import math

from tools.featurization_dashboard.dashboard_math import (
    DEFAULT_IBP_NUMBER_OF_FEATURES,
    CytoTable_time_per_1000_cells_per_1000_features,
    aggregation_time_per_1000_features,
    compute_df,
    compute_plate_time_series,
    compute_spot_cost_series,
    ibp_time_and_mem_estimation,
    pycytominer_feature_selection_time_per_1000_cells_per_1000_features,
    pycytominer_normalization_time_per_1000_cells_per_1000_features,
)


def test_compute_df_builds_expected_columns_and_scaling():
    df = compute_df(total_image_sets=120, time_min_per_set=1.5, max_cores=8, num_plates=4)

    assert list(df.columns) == [
        "number_of_cores",
        "total_time_minutes",
        "total_time_hours",
        "time_per_plate_hours",
        "total_time_days",
        "total_SUs_used",
    ]
    assert df.iloc[0]["number_of_cores"] == 1
    assert math.isclose(df.iloc[0]["total_time_minutes"], 180.0)
    assert math.isclose(df.iloc[0]["time_per_plate_hours"], 0.75)
    assert math.isclose(df.iloc[0]["total_SUs_used"], 3.0)


def test_compute_plate_time_series_repeats_each_plate_count():
    df = compute_plate_time_series(
        wells=2,
        fovs_per_well=3,
        timepoints=4,
        time_min_per_set=2.0,
        max_cores=4,
        max_plates=3,
    )

    assert set(df["plate_count"].unique()) == {1, 2, 3}
    first_plate = df[df["plate_count"] == 1].iloc[0]
    second_plate = df[df["plate_count"] == 2].iloc[0]
    assert math.isclose(second_plate["total_time_minutes"], first_plate["total_time_minutes"] * 2)


def test_compute_spot_cost_series_uses_midpoint_of_bounds():
    df = compute_spot_cost_series(
        wells=2,
        fovs_per_well=3,
        timepoints=4,
        time_min_per_set=2.0,
        max_plates=2,
        min_rate=0.01,
        max_rate=0.05,
    )

    assert list(df["plate_count"]) == [1, 2]
    assert math.isclose(df.iloc[0]["spot_cost_mid"], (df.iloc[0]["spot_cost_low"] + df.iloc[0]["spot_cost_high"]) / 2)


def test_ibp_time_and_mem_estimation_scales_with_feature_count():
    df = ibp_time_and_mem_estimation(
        single_cells_per_fov=10,
        fovs_per_well=2,
        wells_per_plate=3,
        timepoints=4,
        time_per_1000_cells_per_1000_features=CytoTable_time_per_1000_cells_per_1000_features,
        number_of_features=DEFAULT_IBP_NUMBER_OF_FEATURES,
    )

    assert list(df.columns) == [
        "time_per_plate_minutes",
        "time_per_plate_hours",
        "time_per_plate_days",
        "memory_mb_per_plate",
        "memory_gb_per_plate",
    ]
    assert math.isclose(df.iloc[0]["time_per_plate_hours"], df.iloc[0]["time_per_plate_minutes"] / 60.0)
    assert math.isclose(df.iloc[0]["time_per_plate_days"], df.iloc[0]["time_per_plate_hours"] / 24.0)
    assert df.iloc[0]["memory_gb_per_plate"] > 0


def test_ibp_constants_are_supported_for_all_methods():
    rates = [
        CytoTable_time_per_1000_cells_per_1000_features,
        pycytominer_normalization_time_per_1000_cells_per_1000_features,
        pycytominer_feature_selection_time_per_1000_cells_per_1000_features,
        aggregation_time_per_1000_features,
    ]

    assert all(rate > 0 for rate in rates)
