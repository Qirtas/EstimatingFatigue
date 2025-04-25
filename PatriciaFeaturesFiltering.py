import pandas as pd
import os
import glob
import re


def filter_emg_features(features_directory, excluded_sensors):
    """
    Filter out EMG features that contain any of the excluded sensor names.

    Args:
        features_directory: Base directory containing all task feature files
        excluded_sensors: List of sensor names to exclude

    Returns:
        Dictionary with task names as keys and counts of removed features as values
    """
    # Dictionary to store counts of removed features per task
    removed_counts = {}

    # Create regex patterns for matching excluded sensors (case insensitive)
    patterns = [re.compile(f"{sensor}", re.IGNORECASE) for sensor in excluded_sensors]

    # Get list of tasks (subdirectories)
    tasks = [d for d in os.listdir(features_directory)
             if os.path.isdir(os.path.join(features_directory, d))]

    for task in tasks:
        task_dir = os.path.join(features_directory, task)
        csv_files = glob.glob(os.path.join(task_dir, "*.csv"))

        if not csv_files:
            print(f"No CSV files found for task {task}")
            continue

        # Process each CSV file in the task directory
        for file_path in csv_files:
            print(f"Processing {os.path.basename(file_path)} for task {task}...")

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Get original column count
            original_cols = df.shape[1]

            # Identify columns to remove (those that contain any excluded sensor name)
            columns_to_remove = []

            for col in df.columns:
                # Check if column contains any excluded sensor name
                if any(pattern.search(col) for pattern in patterns):
                    columns_to_remove.append(col)

            print(f"Found {len(columns_to_remove)} features to remove from {os.path.basename(file_path)}")

            # Remove the identified columns
            df = df.drop(columns=columns_to_remove)

            # Calculate how many columns were removed
            removed_count = original_cols - df.shape[1]

            # Save the updated DataFrame back to the same file
            df.to_csv(file_path, index=False)

            print(f"Removed {removed_count} features from {os.path.basename(file_path)}")

            # Update the counts dictionary
            removed_counts[task] = removed_counts.get(task, 0) + removed_count

    return removed_counts
