import pandas as pd
import os
import numpy as np


def create_sensor_subsets(importance_file, max_subset_size=10):

    sensor_importance = pd.read_csv(importance_file)

    # Sort by average importance (ensure it's sorted)
    sensor_importance = sensor_importance.sort_values('Avg_Importance', ascending=False).reset_index(drop=True)

    # Create subsets
    subsets = {}
    for i in range(1, min(len(sensor_importance), max_subset_size) + 1):
        subset_sensors = sensor_importance['Sensor'].iloc[:i].tolist()
        subsets[i] = subset_sensors

    return subsets


def load_feature_mapping(mapping_file):

    mapping_df = pd.read_csv(mapping_file)

    # Ensure required columns exist
    required_cols = ['full_feature_name', 'sensor_id']
    if not all(col in mapping_df.columns for col in required_cols):
        raise ValueError(f"Mapping file missing required columns. Expected: {required_cols}")

    return mapping_df


def filter_features_by_sensor_subset(features_df, sensor_subset):

    all_columns = features_df.columns.tolist()

    # Define metadata columns that should always be kept
    metadata_cols = ['Subject', 'Repetition', 'RPE']

    # Find metadata columns in the actual dataset (case-insensitive)
    metadata_cols_in_data = []
    for col in all_columns:
        if any(meta.lower() == col.lower() for meta in metadata_cols):
            metadata_cols_in_data.append(col)

    print(f"  Metadata columns identified: {metadata_cols_in_data}")

    # All other columns are considered features
    feature_cols = [col for col in all_columns if col not in metadata_cols_in_data]
    print(f"  Total feature columns before filtering: {len(feature_cols)}")

    # Create mapping from sensor ID to keywords to match in feature names
    sensor_keywords = {}
    for sensor in sensor_subset:
        if sensor.startswith('EMG_'):
            # For EMG sensors, use the muscle name (e.g., 'tricep' from 'EMG_tricep')
            muscle = sensor.replace('EMG_', '')
            sensor_keywords[sensor] = [muscle]
        elif sensor.startswith('IMU_'):
            # For IMU sensors, use the location (e.g., 'Forearm' from 'IMU_Forearm')
            location = sensor.replace('IMU_', '')
            sensor_keywords[sensor] = [location]

    # For debugging - show which keywords we're looking for
    print(f"  Searching for these keywords in feature names: {sensor_keywords}")

    # Find features matching any of the sensor keywords
    valid_features = []
    matched_features_count = 0

    for feature in feature_cols:
        for sensor, keywords in sensor_keywords.items():
            if any(keyword in feature for keyword in keywords):
                valid_features.append(feature)
                matched_features_count += 1
                break

    # Print matching statistics
    print(f"  Features matched to specified sensors: {matched_features_count} out of {len(feature_cols)}")

    # Keep valid features and metadata columns
    selected_cols = metadata_cols_in_data + valid_features
    filtered_df = features_df[selected_cols]

    return filtered_df


def filter_and_save_dataset(task_file, sensor_subsets, output_dir):

    task_name = os.path.basename(task_file).replace('labeled_features_', '').replace('.csv', '')
    print(f"\nProcessing task: {task_name} from file: {os.path.basename(task_file)}")

    # Create task output directory
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    # Read features file
    features_df = pd.read_csv(task_file)
    total_columns = len(features_df.columns)

    print(f"  Total columns in dataset: {total_columns}")

    # Record feature counts for each subset
    feature_counts = {}

    # Filter and save for each subset
    for subset_size, sensor_subset in sensor_subsets.items():
        print(f"\n  Processing subset {subset_size} with sensors: {sensor_subset}")

        # Filter features
        filtered_df = filter_features_by_sensor_subset(features_df, sensor_subset)

        # Save filtered dataset
        output_file = os.path.join(task_output_dir, f'subset_{subset_size}_sensors.csv')
        filtered_df.to_csv(output_file, index=False)

        # Get feature count (excluding metadata columns)
        feature_count = len(filtered_df.columns)
        feature_counts[subset_size] = feature_count

        print(f"  Saved filtered dataset with {feature_count} columns to: {os.path.basename(output_file)}")

    # Print summary of feature counts for this task
    print(f"\nFeature count summary for task {task_name}:")
    for subset_size, count in feature_counts.items():
        print(f"  Subset {subset_size}: {count} columns")

    return feature_counts


def make_sensor_subsets(tasks_dir, importance_file, output_dir, max_subset_size=10):

    os.makedirs(output_dir, exist_ok=True)

    # Create sensor subsets
    sensor_subsets = create_sensor_subsets(importance_file, max_subset_size)

    # Print subsets for reference
    print("Created sensor subsets:")
    for size, sensors in sensor_subsets.items():
        print(f"Subset {size}: {sensors}")

    # Find all task files
    task_files = [os.path.join(tasks_dir, f) for f in os.listdir(tasks_dir)
                  if f.startswith('labeled_features_') and f.endswith('.csv')]

    if not task_files:
        raise ValueError(f"No task files found in {tasks_dir}")

    # Process each task
    all_task_summaries = {}
    for task_file in task_files:
        feature_counts = filter_and_save_dataset(task_file, sensor_subsets, output_dir)
        task_name = os.path.basename(task_file).replace('labeled_features_', '').replace('.csv', '')
        all_task_summaries[task_name] = feature_counts

    print("\nAll tasks processed successfully!")

    # Create a summary file with information about each subset
    with open(os.path.join(output_dir, 'sensor_subsets_summary.txt'), 'w') as f:
        f.write("Sensor Subsets:\n")
        for size, sensors in sensor_subsets.items():
            f.write(f"Subset {size}: {sensors}\n")

        f.write("\nFeature Count Summary:\n")
        for task, counts in all_task_summaries.items():
            f.write(f"\nTask: {task}\n")
            for subset_size, count in counts.items():
                f.write(f"  Subset {subset_size}: {count} columns\n")
