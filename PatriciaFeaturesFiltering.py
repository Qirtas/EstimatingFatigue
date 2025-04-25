import pandas as pd
import os
import glob
import re

def filter_emg_features(features_directory, excluded_sensors):
    removed_counts = {}

    patterns = [re.compile(f"{sensor}", re.IGNORECASE) for sensor in excluded_sensors]

    tasks = [d for d in os.listdir(features_directory)
             if os.path.isdir(os.path.join(features_directory, d))]

    for task in tasks:
        task_dir = os.path.join(features_directory, task)
        csv_files = glob.glob(os.path.join(task_dir, "*.csv"))

        if not csv_files:
            print(f"No CSV files found for task {task}")
            continue

        for file_path in csv_files:
            print(f"Processing {os.path.basename(file_path)} for task {task}...")

            df = pd.read_csv(file_path)

            original_cols = df.shape[1]

            columns_to_remove = []

            for col in df.columns:
                if any(pattern.search(col) for pattern in patterns):
                    columns_to_remove.append(col)

            print(f"Found {len(columns_to_remove)} features to remove from {os.path.basename(file_path)}")

            df = df.drop(columns=columns_to_remove)

            removed_count = original_cols - df.shape[1]

            df.to_csv(file_path, index=False)

            print(f"Removed {removed_count} features from {os.path.basename(file_path)}")

            removed_counts[task] = removed_counts.get(task, 0) + removed_count

    return removed_counts
