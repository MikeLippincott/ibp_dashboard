import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tools.featurization_dashboard.dashboard_math import (
    CytoTable_time_per_1000_cells_per_1000_features,
    aggregation_time_per_1000_features,
    compute_df as compute_df_core,
    compute_plate_time_series,
    compute_spot_cost_series,
    ibp_time_and_mem_estimation,
    pycytominer_feature_selection_time_per_1000_cells_per_1000_features,
    pycytominer_normalization_time_per_1000_cells_per_1000_features,
)

st.set_page_config(layout="wide", page_title="Featurization Time Dashboard (Plotly)")

st.title("Featurization Time / Compute Planning Dashboard")

# Sidebar inputs
st.sidebar.header("Dataset parameters")
PLATES = st.sidebar.number_input("Plates", min_value=1, value=10, step=1)
WELLS = st.sidebar.number_input("Wells per plate", min_value=1, value=60, step=1)
FOVS_PER_WELL = st.sidebar.number_input("FOVs per well", min_value=1, value=4, step=1)
SINGLE_CELLS_PER_FOV = st.sidebar.number_input("Single cells per FOV\n(for IBP estimates)", min_value=1, value=1000, step=1)
TIMEPOINTS = st.sidebar.number_input("Timepoints\n(set to 1 for static data)", min_value=1, value=100, step=1)
TIME_TO_FEATURIZE_ONE_IMAGE_SET = st.sidebar.number_input(
    "Time to featurize one image set (minutes)", min_value=0.1, value=1.5, step=0.1, format="%.1f"
)
NUMBER_OF_FEATURES = st.sidebar.number_input("Number of features per image set (for IBP estimates)", min_value=1, value=5000, step=1)

st.sidebar.header("Compute parameters")
MAX_WORKERS_AVAILABLE = st.sidebar.number_input(
    "Max local workers available", min_value=1, value=128, step=1
)
AWS_WORKERS_SOFT_MAX = st.sidebar.number_input(
    "AWS soft max (workers)", min_value=1, value=1440, step=1
)
AWS_WORKERS_HARD_MAX = st.sidebar.number_input(
    "AWS hard max (workers)", min_value=1, value=20000, step=1
)

st.sidebar.header("Plot options")
primary_units = st.sidebar.selectbox("First plot y-axis units", ["minutes", "hours", "days"], index=1)
include_serial_time = st.sidebar.checkbox("Include serial time (1 core)", value=False)
first_plot_y_column = {
    "minutes": "total_time_minutes",
    "hours": "total_time_hours",
    "days": "total_time_days",
}[primary_units]
plot_choices = st.sidebar.multiselect(
    "Plots to show",
    options=["Total Hours", "Time per Plate", "Spot Cost"],
    default=["Total Hours", "Time per Plate", "Spot Cost"],
)
annotate_cores = st.sidebar.multiselect(
    "Annotate these core counts (show in textbox)", [1, 16, 32, 64, 128, AWS_WORKERS_SOFT_MAX, AWS_WORKERS_HARD_MAX], default=[1, 16, 32, 64, 128]
)
cores_display_limit = st.sidebar.slider("Number of core samples to compute", min_value=10, max_value=20000, value=500, step=10)

# Derived values
WELL_FOVS = WELLS * FOVS_PER_WELL * PLATES
TOTAL_IMAGE_SETS = WELL_FOVS * TIMEPOINTS

st.markdown(f"**Total image sets:** {TOTAL_IMAGE_SETS:,}  ")


@st.cache_data
def compute_df(total_image_sets, time_min_per_set, max_cores, num_plates):
    return compute_df_core(total_image_sets, time_min_per_set, max_cores, num_plates)


cores_num = max(AWS_WORKERS_HARD_MAX, MAX_WORKERS_AVAILABLE, cores_display_limit)
df = compute_df(TOTAL_IMAGE_SETS, TIME_TO_FEATURIZE_ONE_IMAGE_SET, cores_num, PLATES)

# Sample for plotting
if len(df) > cores_display_limit:
    step = max(1, len(df) // cores_display_limit)
    df_plot = df.iloc[::step].copy()
else:
    df_plot = df.copy()

st.subheader("Interactive plots")
cols = st.columns((2, 1))

with cols[0]:
    # Total hours plot
    if "Total Hours" in plot_choices:
        plate_df = compute_plate_time_series(WELLS, FOVS_PER_WELL, TIMEPOINTS, TIME_TO_FEATURIZE_ONE_IMAGE_SET, cores_num, PLATES)
        if not include_serial_time:
            plate_df = plate_df[plate_df["number_of_cores"] > 1]
        fig_hours = go.Figure()
        palette = px.colors.qualitative.Set2 + px.colors.qualitative.Dark24
        for plate_count in range(1, PLATES + 1):
            plate_data = plate_df[plate_df["plate_count"] == plate_count]
            if plate_data.empty:
                continue
            fig_hours.add_trace(go.Scatter(
                x=plate_data["number_of_cores"],
                y=plate_data[first_plot_y_column],
                mode="lines+markers",
                name="",
                line=dict(color=palette[(plate_count - 1) % len(palette)], width=3),
                marker=dict(size=5),
                showlegend=False,
            ))
        y_title = primary_units.capitalize()
        fig_hours.update_layout(
            title=f"Total Time ({primary_units}) — {WELL_FOVS:,} image sets per plate",
            xaxis_title="Cores",
            yaxis_title=y_title,
        )
        fig_hours.update_yaxes(type="log")
        fig_hours.update_layout(showlegend=False)
        # add vertical lines for key points
        key_points = sorted(set([1, 16, 32, 64, 128, MAX_WORKERS_AVAILABLE, AWS_WORKERS_SOFT_MAX, AWS_WORKERS_HARD_MAX]))
        if not include_serial_time:
            key_points = [kp for kp in key_points if kp != 1]
        for kp in key_points:
            if kp in df['number_of_cores'].values:
                y = float(df.loc[df['number_of_cores'] == kp, first_plot_y_column].values[0])
                fig_hours.add_vline(x=kp, line=dict(color='gray', dash='dash'), opacity=0.4)
                fig_hours.add_annotation(x=kp, y=y, text=f"{kp}", showarrow=False, yshift=10)
        st.plotly_chart(fig_hours, use_container_width=True)

    if "Time per Plate" in plot_choices:
        fig_plate = px.line(df_plot, x="number_of_cores", y="time_per_plate_hours", markers=True,
                            labels={"number_of_cores": "Cores", "time_per_plate_hours": "Hours per plate"},
                            title=f"Time per Plate (hours) — {PLATES:,} plates")
        fig_plate.update_yaxes(type="log")
        fig_plate.update_layout(showlegend=False)
        st.plotly_chart(fig_plate, use_container_width=True)

    if "Spot Cost" in plot_choices:
        cost_df = compute_spot_cost_series(WELLS, FOVS_PER_WELL, TIMEPOINTS, TIME_TO_FEATURIZE_ONE_IMAGE_SET, PLATES, 0.006, 0.04)
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=cost_df["cpu_hours_requested"],
            y=cost_df["spot_cost_mid"],
            mode="lines+markers",
            line=dict(color="#7c3aed", width=3),
            marker=dict(size=6),
            showlegend=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=cost_df["spot_cost_high"] - cost_df["spot_cost_mid"],
                arrayminus=cost_df["spot_cost_mid"] - cost_df["spot_cost_low"],
                visible=True,
                thickness=1.5,
                width=0,
                color="#7c3aed",
            ),
        ))
        fig_cost.update_layout(
            title=f"Spot Cost Estimate — ${0.006:.3f} to ${0.04:.2f} per CPU-hour",
            xaxis_title="Total CPU-hours requested for all image-sets",
            yaxis_title="Cost (USD)",
            yaxis=dict(tickprefix="$", tickformat=",.2f"),
            showlegend=False,
        )
        st.plotly_chart(fig_cost, use_container_width=True)
with cols[1]:
    st.subheader("Selected values")
    st.write(f"Selected core annotations and AWS max values ({primary_units})")
    table_rows = []
    def add_table_row(label, core_value):
        if not include_serial_time and core_value == 1:
            return
        if core_value in df['number_of_cores'].values:
            value = float(df.loc[df['number_of_cores'] == core_value, first_plot_y_column].values[0])
            table_rows.append({"label": label, "value": round(value, 1)})
    for c in annotate_cores:
        add_table_row(f"Core {int(c)}", c)
    add_table_row("AWS soft max", AWS_WORKERS_SOFT_MAX)
    add_table_row("AWS hard max", AWS_WORKERS_HARD_MAX)
    if table_rows:
        st.table(pd.DataFrame(table_rows))
    else:
        st.info("No annotated cores present in computed range.")

csv = df.to_csv(index=False)
st.download_button(label='Download full CSV', data=csv, file_name='featurization_times.csv', mime='text/csv')



# add ibp section here
st.subheader("Image-based profiling (IBP) estimations")
st.markdown("""Note this is not an accurate section more benchmarking needs to be performed first.\nThis section provides rough estimates for how long it would take to featurize a dataset using image-based profiling methods like CellProfiler or deep learning-based feature extraction. These are very rough estimates and can vary widely based on the specific methods, hardware, and dataset characteristics. We make a key assumption here that the number of plates never exceeds the available number of cores/machines, thus all plates can be run in parallel. Adjust the number of single cells per FOV, number of features and timepoints to see how it impacts the estimates.""")

get_interpolated_time_and_memory_usage(new_sample_number, new_feature_number)


# plot for IBP time per plate for CytoTable based on the number of single-cells
cytotable_df = ibp_time_and_mem_estimation(SINGLE_CELLS_PER_FOV, FOVS_PER_WELL, WELLS, TIMEPOINTS, CytoTable_time_per_1000_cells_per_1000_features, number_of_features=NUMBER_OF_FEATURES)
pycytominer_norm_df = ibp_time_and_mem_estimation(SINGLE_CELLS_PER_FOV, FOVS_PER_WELL, WELLS, TIMEPOINTS, pycytominer_normalization_time_per_1000_cells_per_1000_features, number_of_features=NUMBER_OF_FEATURES)
pycytominer_fs_df = ibp_time_and_mem_estimation(SINGLE_CELLS_PER_FOV, FOVS_PER_WELL, WELLS, TIMEPOINTS, pycytominer_feature_selection_time_per_1000_cells_per_1000_features, number_of_features=NUMBER_OF_FEATURES)
pycytominer_agg_df = ibp_time_and_mem_estimation(SINGLE_CELLS_PER_FOV, FOVS_PER_WELL, WELLS, TIMEPOINTS, aggregation_time_per_1000_features, number_of_features=NUMBER_OF_FEATURES)
cytotable_df["method"] = "CytoTable Harmonization"
pycytominer_norm_df["method"] = "pycytominer Normalization"
pycytominer_fs_df["method"] = "pycytominer Feature Selection"
pycytominer_agg_df["method"] = "pycytominer Aggregation"
ibp_combined_df = pd.concat([cytotable_df, pycytominer_norm_df, pycytominer_fs_df, pycytominer_agg_df], ignore_index=True)

ibp_time_column = {
    "minutes": "time_per_plate_minutes",
    "hours": "time_per_plate_hours",
    "days": "time_per_plate_days",
}[primary_units]
ibp_time_label = primary_units.capitalize()

# plot a bar plot for time per plate for each method
fig_ibp = px.bar(ibp_combined_df, x="method", y=ibp_time_column,
                 labels={ibp_time_column: f"Estimated {ibp_time_label} per plate", "method": "IBP method"},
                 title=f"Estimated Time per Plate for IBP Methods ({ibp_time_label}; based on {SINGLE_CELLS_PER_FOV} single cells per FOV)")
st.plotly_chart(fig_ibp, use_container_width=True)

# plot a bar plot for time per plate for each method
fig_mem_ibp = px.bar(ibp_combined_df, x="method", y="memory_gb_per_plate",
                     labels={"memory_gb_per_plate": "Estimated Memory (GB) per plate", "method": "IBP method"},
                     title=f"Estimated Memory per Plate for IBP Methods (GB; based on {SINGLE_CELLS_PER_FOV} single cells per FOV)")
st.plotly_chart(fig_mem_ibp, use_container_width=True)
