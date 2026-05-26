import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from featurization_dashboard.dashboard_math import (
    CytoTable_time_per_1000_cells_per_1000_features,
    aggregation_time_per_1000_features,
    compute_df as compute_df_core,
    compute_parallelized_feature_time_series,
    compute_plate_time_series,
    compute_spot_cost_series,
    call_metric_interpolation_given_parameters,
    pycytominer_feature_selection_time_per_1000_cells_per_1000_features,
    pycytominer_normalization_time_per_1000_cells_per_1000_features,
)

st.set_page_config(layout="wide", page_title="Featurization Time Dashboard (Plotly)")

st.title("Featurization Time / Compute Planning Dashboard")

# Sidebar inputs
st.sidebar.header("Dataset parameters")
PLATES = st.sidebar.number_input("Plates", min_value=1, value=2, step=1)
WELLS = st.sidebar.number_input("Wells per plate", min_value=1, value=308, step=1)
FOVS_PER_WELL = st.sidebar.number_input("FOVs per well", min_value=1, value=4, step=1)
SINGLE_CELLS_PER_FOV = st.sidebar.number_input("Single cells per FOV\n(for IBP estimates)", min_value=1, value=1000, step=1)
TIMEPOINTS = st.sidebar.number_input("Timepoints\n(set to 1 for static data)", min_value=1, value=2, step=1)
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

st.sidebar.header("Plot options")
primary_units = st.sidebar.selectbox("First plot y-axis units", ["minutes", "hours", "days"], index=1)
include_serial_time = st.sidebar.checkbox("Include serial time (1 core)", value=False)
parallel_method = st.sidebar.selectbox(
    "Parallelization method",
    options=["plate_well_fov_time", "well_fov", "well", "plate"],
    index=0,
)
first_plot_y_column = {
    "minutes": "total_time_minutes",
    "hours": "total_time_hours",
    "days": "total_time_days",
}[primary_units]
plot_choices = st.sidebar.multiselect(
    "Plots to show",
    options=["Total Hours", "Spot Cost"],
    default=["Total Hours", "Spot Cost"],
)
annotate_cores = st.sidebar.multiselect(
    "Annotate these core counts (show in textbox)", [1, 16, 32, 64, 128, AWS_WORKERS_SOFT_MAX], default=[1, 16, 32, 64, 128]
)
cores_display_limit = st.sidebar.slider("Number of core samples to compute", min_value=10, max_value=20000, value=500, step=10)

# Derived values
WELL_FOVS = WELLS * FOVS_PER_WELL * PLATES
TOTAL_IMAGE_SETS = WELL_FOVS * TIMEPOINTS

st.markdown(f"**Total image sets:** {TOTAL_IMAGE_SETS:,}  ")


@st.cache_data
def compute_df(total_image_sets, time_min_per_set, max_cores, num_plates):
    return compute_df_core(total_image_sets, time_min_per_set, max_cores, num_plates)


cores_num = max(AWS_WORKERS_SOFT_MAX, MAX_WORKERS_AVAILABLE, cores_display_limit)
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
        # Replace Total Hours with the selected parallelization method curve
        parallel_df_total = compute_parallelized_feature_time_series(
            plates=PLATES,
            wells_per_plate=WELLS,
            fovs_per_well=FOVS_PER_WELL,
            timepoints=TIMEPOINTS,
            time_min_per_image_set=TIME_TO_FEATURIZE_ONE_IMAGE_SET,
            max_cores=cores_num,
        )
        if not include_serial_time:
            parallel_df_total = parallel_df_total[parallel_df_total["number_of_cores"] > 1]

        # filter for the selected method
        if parallel_method:
            parallel_df_total = parallel_df_total[parallel_df_total["parallelization_level"] == parallel_method]

        if parallel_df_total.empty:
            st.info("No data to plot for selected parallelization method.")
        else:
            ycol = {
                "minutes": "total_time_minutes",
                "hours": "total_time_hours",
                "days": "total_time_days",
            }[primary_units]

            fig_hours = go.Figure()
            fig_hours.add_trace(go.Scatter(
                x=parallel_df_total["number_of_cores"],
                y=parallel_df_total[ycol],
                mode="lines+markers",
                name=parallel_method,
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=6),
            ))

            fig_hours.update_layout(
                title=f"Total Time ({primary_units}) — {parallel_method}",
                xaxis_title="Cores",
                yaxis_title=primary_units.capitalize(),
            )
            fig_hours.update_yaxes(type="log")
            # add vertical lines for key points
            key_points = sorted(set([1, 16, 32, 64, 128, MAX_WORKERS_AVAILABLE, AWS_WORKERS_SOFT_MAX]))
            if not include_serial_time:
                key_points = [kp for kp in key_points if kp != 1]
            for kp in key_points:
                if kp in df['number_of_cores'].values:
                    y = float(df.loc[df['number_of_cores'] == kp, first_plot_y_column].values[0])
                    fig_hours.add_vline(x=kp, line=dict(color='gray', dash='dash'), opacity=0.4)
                    fig_hours.add_annotation(x=kp, y=y, text=f"{kp}", showarrow=False, yshift=10)

            st.plotly_chart(fig_hours, use_container_width=True, key=f"total_hours_{parallel_method}_{primary_units}_{cores_num}")

    # 'Time per Plate' plot removed per user request

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

    # parallelization-level plot removed (merged into Total Hours per-method curve)
with cols[1]:
    st.subheader("Selected values")
    st.write(f"Selected core annotations and AWS max values ({primary_units})")

    # Show merged table by selected parallelization method
    if parallel_method:
        # build pivoted table from parallel_df
        parallel_df = compute_parallelized_feature_time_series(
            plates=PLATES,
            wells_per_plate=WELLS,
            fovs_per_well=FOVS_PER_WELL,
            timepoints=TIMEPOINTS,
            time_min_per_image_set=TIME_TO_FEATURIZE_ONE_IMAGE_SET,
            max_cores=cores_num,
        )
        if not include_serial_time:
            parallel_df = parallel_df[parallel_df["number_of_cores"] > 1]

        # apply same filtering as the plot (single method)
        if parallel_method:
            parallel_df = parallel_df[parallel_df["parallelization_level"] == parallel_method]
        else:
            parallel_df = parallel_df.iloc[0:0]

        unit_col = {
            "minutes": "total_time_minutes",
            "hours": "total_time_hours",
            "days": "total_time_days",
        }[primary_units]
        # prepare display rows for annotated cores for the selected method
        series = parallel_df.set_index("number_of_cores")[unit_col]
        display_rows = []
        for c in annotate_cores + [AWS_WORKERS_SOFT_MAX]:
            if not include_serial_time and c == 1:
                continue
            if c in series.index:
                v = series.loc[c]
                display_rows.append({"cores": int(c), parallel_method: round(float(v), 2)})

        if display_rows:
            display_df = pd.DataFrame(display_rows).set_index("cores")
            st.dataframe(display_df, key=f"parallel_table_{parallel_method}_{primary_units}_{cores_num}")
        else:
            st.info("No annotated cores present in computed range.")
    else:
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
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), key=f"summary_table_{primary_units}_{cores_num}")
        else:
            st.info("No annotated cores present in computed range.")

csv = df.to_csv(index=False)
st.download_button(label='Download full CSV', data=csv, file_name='featurization_times.csv', mime='text/csv')



# add ibp section here
st.subheader("Image-based profiling (IBP) estimations")
st.markdown("""Note this is not an accurate section more benchmarking needs to be performed first.\nThis section provides rough estimates for how long it would take to featurize a dataset using image-based profiling methods like CellProfiler or deep learning-based feature extraction. These are very rough estimates and can vary widely based on the specific methods, hardware, and dataset characteristics. We make a key assumption here that the number of plates never exceeds the available number of cores/machines, thus all plates can be run in parallel. Adjust the number of single cells per FOV, number of features and timepoints to see how it impacts the estimates.""")


# plot for IBP time per plate for CytoTable based on the number of single-cells
# cytotable_df = ibp_time_and_mem_estimation(SINGLE_CELLS_PER_FOV, FOVS_PER_WELL, WELLS, TIMEPOINTS, CytoTable_time_per_1000_cells_per_1000_features, number_of_features=NUMBER_OF_FEATURES)
try:
    pycytominer_df = call_metric_interpolation_given_parameters(
        number_of_single_cells_per_fov=SINGLE_CELLS_PER_FOV,
        number_of_features=NUMBER_OF_FEATURES,
        fovs_per_well=FOVS_PER_WELL,
        wells_per_plate=WELLS,
        timepoints=TIMEPOINTS,
    )

    pycytominer_df.rename(columns={
        "interpolated_time_seconds": "time_seconds"
    }, inplace=True)
    pycytominer_df['time_minutes'] = pycytominer_df['time_seconds'] / 60.0
    pycytominer_df['time_hours'] = pycytominer_df['time_minutes'] / 60.0
    pycytominer_df['time_days'] = pycytominer_df['time_hours'] / 24.0

    ibp_time_column = {
        "seconds": "time_seconds",
        "minutes": "time_minutes",
        "hours": "time_hours",
        "days": "time_days",
    }[primary_units]
    ibp_time_label = primary_units.capitalize()

    # plot a bar plot for time per plate for each method
    fig_ibp = px.bar(pycytominer_df, x="process_name", y=ibp_time_column,
                     labels={ibp_time_column: f"Estimated {ibp_time_label} per plate", "method": "IBP method"},
                     title=f"Estimated Time per Plate for IBP Methods ({ibp_time_label}; based on {SINGLE_CELLS_PER_FOV} single cells per FOV)")
    st.plotly_chart(fig_ibp, use_container_width=True)

    # plot a bar plot for memory per plate for each method
    fig_mem_ibp = px.bar(pycytominer_df, x="process_name", y="interpolated_memory_GB",
                         labels={"interpolated_memory_GB": "Estimated Memory (GB) per plate", "method": "IBP method"},
                         title=f"Estimated Memory per Plate for IBP Methods (GB; based on {SINGLE_CELLS_PER_FOV} single cells per FOV)")
    st.plotly_chart(fig_mem_ibp, use_container_width=True)
except Exception as exc:
    st.error("IBP estimation data could not be loaded in this deployment.")
    st.caption(f"Details: {exc}")
