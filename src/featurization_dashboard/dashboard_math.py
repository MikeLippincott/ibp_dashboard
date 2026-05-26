import pathlib

import numpy as np
import pandas as pd


DEFAULT_IBP_NUMBER_OF_FEATURES = 5000

CytoTable_time_per_1000_cells_per_1000_features = 0.1
pycytominer_normalization_time_per_1000_cells_per_1000_features = 0.1
pycytominer_feature_selection_time_per_1000_cells_per_1000_features = 0.1
aggregation_time_per_1000_features = 0.1


def compute_df(total_image_sets, time_min_per_set, max_cores, num_plates):
    small = np.arange(1, min(max_cores, 256) + 1)
    large = np.unique(np.linspace(256, max_cores, num=min(max_cores - 255, 1000)).astype(int)) if max_cores > 256 else np.array([])
    numbers = np.unique(np.concatenate([small, large]))
    numbers = numbers[numbers >= 1]
    total_minutes = (total_image_sets * time_min_per_set) / numbers
    df = pd.DataFrame({
        "number_of_cores": numbers,
        "total_time_minutes": total_minutes,
    })
    df["total_time_hours"] = df["total_time_minutes"] / 60.0
    df["time_per_plate_hours"] = df["total_time_hours"] / num_plates
    df["total_time_days"] = df["total_time_hours"] / 24.0
    df["total_SUs_used"] = df["number_of_cores"] * df["total_time_hours"]
    return df


def compute_plate_time_series(wells, fovs_per_well, timepoints, time_min_per_set, max_cores, max_plates):
    numbers = compute_df(1, time_min_per_set, max_cores, 1)["number_of_cores"].to_numpy()
    base_image_sets_per_plate = wells * fovs_per_well * timepoints
    rows = []
    for plate_count in range(1, max_plates + 1):
        total_minutes = (base_image_sets_per_plate * plate_count * time_min_per_set) / numbers
        rows.append(pd.DataFrame({
            "plate_count": plate_count,
            "number_of_cores": numbers,
            "total_time_minutes": total_minutes,
            "total_time_hours": total_minutes / 60.0,
            "total_time_days": total_minutes / 60.0 / 24.0,
        }))
    return pd.concat(rows, ignore_index=True)


def compute_spot_cost_series(wells, fovs_per_well, timepoints, time_min_per_set, max_plates, min_rate, max_rate):
    base_cpu_hours_per_plate = (wells * fovs_per_well * timepoints * time_min_per_set) / 60.0
    plate_counts = np.arange(1, max_plates + 1)
    cpu_hours = base_cpu_hours_per_plate * plate_counts
    low_cost = cpu_hours * min_rate
    high_cost = cpu_hours * max_rate
    mid_cost = (low_cost + high_cost) / 2.0
    return pd.DataFrame({
        "plate_count": plate_counts,
        "cpu_hours_requested": cpu_hours,
        "spot_cost_low": low_cost,
        "spot_cost_high": high_cost,
        "spot_cost_mid": mid_cost,
    })


def compute_parallelized_feature_time_series(
    plates,
    wells_per_plate,
    fovs_per_well,
    timepoints,
    time_min_per_image_set,
    max_cores,
):
    """Estimate total wall-clock time across different parallelization levels.

    Parallelization levels are based on the granularity of independent tasks:
    - plate_well_fov_time: one task per image set
    - well_fov: one task per well/fov across all timepoints
    - well: one task per well across all fovs/timepoints
    - plate: one task per plate
    """

    numbers = compute_df(1, time_min_per_image_set, max_cores, 1)["number_of_cores"].to_numpy()
    levels = {
        "plate_well_fov_time": {
            "task_count": plates * wells_per_plate * fovs_per_well * timepoints,
            "task_duration_minutes": time_min_per_image_set,
        },
        "well_fov": {
            "task_count": plates * wells_per_plate * fovs_per_well,
            "task_duration_minutes": timepoints * time_min_per_image_set,
        },
        "well": {
            "task_count": plates * wells_per_plate,
            "task_duration_minutes": fovs_per_well * timepoints * time_min_per_image_set,
        },
        "plate": {
            "task_count": plates,
            "task_duration_minutes": wells_per_plate * fovs_per_well * timepoints * time_min_per_image_set,
        },
    }

    rows = []
    for level_name, level_meta in levels.items():
        task_count = int(level_meta["task_count"])
        task_duration_minutes = float(level_meta["task_duration_minutes"])
        effective_workers = np.minimum(numbers, task_count)
        total_minutes = (task_count / effective_workers) * task_duration_minutes

        rows.append(pd.DataFrame({
            "parallelization_level": level_name,
            "number_of_cores": numbers,
            "task_count": task_count,
            "task_duration_minutes": task_duration_minutes,
            "effective_workers": effective_workers,
            "total_time_minutes": total_minutes,
            "total_time_hours": total_minutes / 60.0,
            "total_time_days": total_minutes / 60.0 / 24.0,
        }))

    return pd.concat(rows, ignore_index=True)


def _fit_linear_surface(points, values, new_sample_number, new_feature_number):
    design_matrix = np.column_stack([points, np.ones(len(points))])
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, values, rcond=None)
    return float(np.dot([new_sample_number, new_feature_number, 1.0], coefficients))

def get_interpolated_time_and_memory_usage(new_sample_number, new_feature_number):

    # relative to the root of the repository
    df_path = pathlib.Path("./cytomining_benchmarking/profiling_results.parquet").resolve()
    if not df_path.exists():
        raise FileNotFoundError(f"Profiling results parquet file not found at {df_path}")
    df = pd.read_parquet(df_path)
    processes = df["process_name"].dropna().unique()
    interpolated_results = []
    for process in processes:
        subset = df[df["process_name"] == process]
        points = subset[["number_of_samples", "number_of_features"]].values
        elapsed_time = subset["elapsed_time"].values
        peak_memory = subset["peak_memory_usage_GB"].values
        # interpolate if within the bounds of the data
        # otherwise extrapolate but with a warning
        if (new_sample_number < points[:, 0].min() or new_sample_number > points[:, 0].max() or
            new_feature_number < points[:, 1].min() or new_feature_number > points[:, 1].max()):
            print(f"Warning: Extrapolating for process {process} at sample number {new_sample_number} and feature number {new_feature_number}")
        else:
            pass

        interpolated_time = np.round(
            _fit_linear_surface(points, elapsed_time, new_sample_number, new_feature_number),
            2,
        )
        interpolated_memory = np.round(
            _fit_linear_surface(points, peak_memory, new_sample_number, new_feature_number),
            2,
        )
        interpolated_results.append({
            "process_name": process,
            "interpolated_time_seconds": interpolated_time,
            "interpolated_memory_GB": interpolated_memory,
            "number_of_samples": new_sample_number,
            "number_of_features": new_feature_number,
            "matrix_size": new_sample_number * new_feature_number
        })
    interpolated_df = pd.DataFrame(interpolated_results)
    return interpolated_df


def call_metric_interpolation_given_parameters(
    **kwargs
) -> pd.DataFrame:
    number_of_single_cells_per_fov = kwargs.get("number_of_single_cells_per_fov")
    fovs_per_well = kwargs.get("fovs_per_well")
    wells_per_plate = kwargs.get("wells_per_plate")
    number_of_timepoints = kwargs.get("timepoints")
    number_of_features = kwargs.get("number_of_features")


    total_number_of_samples_per_plate = (
        number_of_single_cells_per_fov *
        fovs_per_well *
        wells_per_plate *
        number_of_timepoints
    )

    return get_interpolated_time_and_memory_usage(
        new_sample_number=total_number_of_samples_per_plate,
        new_feature_number=number_of_features,
    )


def ibp_time_and_mem_estimation(
    single_cells_per_fov,
    fovs_per_well,
    wells_per_plate,
    timepoints,
    time_per_1000_cells_per_1000_features,
    number_of_features=DEFAULT_IBP_NUMBER_OF_FEATURES,
) -> pd.DataFrame:
    """Backwards-compatible IBP estimate wrapper used by the tests.

    The current dashboard uses the interpolation-based path, but the public
    API still expects a per-plate summary with minutes, hours, days, and memory.
    """

    interpolated_df = call_metric_interpolation_given_parameters(
        number_of_single_cells_per_fov=single_cells_per_fov,
        number_of_features=number_of_features,
        fovs_per_well=fovs_per_well,
        wells_per_plate=wells_per_plate,
        timepoints=timepoints,
    )

    time_per_plate_minutes = interpolated_df["interpolated_time_seconds"] / 60.0
    memory_gb_per_plate = interpolated_df["interpolated_memory_GB"]

    result_df = pd.DataFrame({
        "time_per_plate_minutes": time_per_plate_minutes,
        "time_per_plate_hours": time_per_plate_minutes / 60.0,
        "time_per_plate_days": time_per_plate_minutes / 60.0 / 24.0,
        "memory_mb_per_plate": memory_gb_per_plate * 1024.0,
        "memory_gb_per_plate": memory_gb_per_plate,
    })
    return result_df
