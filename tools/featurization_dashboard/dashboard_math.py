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


def ibp_time_and_mem_estimation(
    single_cells_per_fov,
    fovs_per_well,
    wells_per_plate,
    timepoints,
    time_per_1000_cells_per_1000_features,
    number_of_features=DEFAULT_IBP_NUMBER_OF_FEATURES,
):
    total_single_cells_per_plate = single_cells_per_fov * fovs_per_well * timepoints * wells_per_plate
    matrix_size = total_single_cells_per_plate * number_of_features
    time_per_plate_minutes = (matrix_size / 1000 / 1000) * time_per_1000_cells_per_1000_features
    memory_mb_per_plate = (matrix_size * 8) / (1024 * 1024)
    memory_gb_per_plate = memory_mb_per_plate / 1024.0
    return pd.DataFrame({
        "time_per_plate_minutes": [time_per_plate_minutes],
        "time_per_plate_hours": [time_per_plate_minutes / 60.0],
        "time_per_plate_days": [time_per_plate_minutes / 60.0 / 24.0],
        "memory_mb_per_plate": [memory_mb_per_plate],
        "memory_gb_per_plate": [memory_gb_per_plate],
    })
