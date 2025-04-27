import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob


def analyze_sensor_importance(feature_mapping_path, feature_importance_dir):

    try:
        mapping_df = pd.read_csv(feature_mapping_path)
        print(f"Loaded feature mapping with {len(mapping_df)} features")
    except Exception as e:
        print(f"Error loading feature mapping file: {e}")
        return {}


    feature_to_sensor = {}
    for _, row in mapping_df.iterrows():
        feature_to_sensor[row['original_feature']] = {
            'sensor_id': row['sensor_id'],
            'sensor_type': row['sensor_type'],
            'location': row['location'],
            'axis': row['axis'],
            'subtype': row['subtype'],
            'feature_type': row['feature_type']
        }
        # Also add the full feature name as a potential key
        feature_to_sensor[row['full_feature_name']] = {
            'sensor_id': row['sensor_id'],
            'sensor_type': row['sensor_type'],
            'location': row['location'],
            'axis': row['axis'],
            'subtype': row['subtype'],
            'feature_type': row['feature_type']
        }

    importance_files = [f for f in os.listdir(feature_importance_dir)
                        if f.startswith('feature_importance_') and f.endswith('.csv')]

    if not importance_files:
        print(f"No feature importance files found in {feature_importance_dir}")
        return {}

    print(f"Found {len(importance_files)} feature importance files")

    results = {}

    for file in importance_files:
        task = file.replace('feature_importance_', '').replace('.csv', '')

        file_path = os.path.join(feature_importance_dir, file)
        try:
            importance_df = pd.read_csv(file_path)
            print(f"Processing {file} with {len(importance_df)} features")

            if len(importance_df.columns) == 1:
                # File doesn't have proper headers - try to parse manually
                # Assuming format: Feature Importance feature_name importance_value
                new_df = pd.DataFrame(columns=['feature', 'importance'])
                for _, row in importance_df.iterrows():
                    content = row[0].split()
                    if len(content) >= 2:  # At least feature name and value
                        # The last element should be the importance value
                        try:
                            importance_val = float(content[-1])
                            # Everything before the last element is the feature name
                            feature_name = ' '.join(content[:-1])
                            new_df = new_df.append({
                                'feature': feature_name.strip(),
                                'importance': importance_val
                            }, ignore_index=True)
                        except:
                            print(f"Could not parse row: {row[0]}")

                importance_df = new_df

            # Ensure we have necessary columns
            if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
                # Try to detect columns based on content
                for col in importance_df.columns:
                    try:
                        float(importance_df[col].iloc[0])
                        importance_col = col
                    except:
                        feature_col = col

                importance_df = importance_df.rename(columns={
                    feature_col: 'feature',
                    importance_col: 'importance'
                })

            # Normalize importance scores (0-1)
            if importance_df['importance'].max() > 0:
                importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].max()
            else:
                importance_df['importance_normalized'] = 0

            # Add sensor mapping to importance df
            sensor_data = []
            unknown_features = []
            for _, row in importance_df.iterrows():
                feature = row['feature']
                sensor_info = feature_to_sensor.get(feature, {})
                if feature not in feature_to_sensor and not any(
                        feature in key or key in feature for key in feature_to_sensor):
                    unknown_features.append(feature)

                if not sensor_info:
                    # Try to find a partial match - the mapping might be more specific than importance file
                    for key in feature_to_sensor:
                        if feature in key or key in feature:
                            sensor_info = feature_to_sensor[key]
                            break

                sensor_data.append({
                    'feature': feature,
                    'importance': row['importance'],
                    'importance_normalized': row['importance_normalized'],
                    'sensor_id': sensor_info.get('sensor_id', 'Unknown'),
                    'sensor_type': sensor_info.get('sensor_type', 'Unknown'),
                    'location': sensor_info.get('location', 'Unknown'),
                    'axis': sensor_info.get('axis', 'Unknown'),
                    'subtype': sensor_info.get('subtype', 'Unknown'),
                    'feature_type': sensor_info.get('feature_type', 'Unknown')
                })

            if unknown_features:
                print(f"Warning: {len(unknown_features)} features in {file} couldn't be mapped to sensors:")
                for feature in unknown_features[:10]:  # Show first 10 examples
                    print(f"  - {feature}")
                if len(unknown_features) > 10:
                    print(f"  - ... and {len(unknown_features) - 10} more")

            # Create a new dataframe with sensor information
            sensor_importance_df = pd.DataFrame(sensor_data)

            # Calculate aggregated importance by sensor
            sensor_agg = sensor_importance_df.groupby('sensor_id').agg({
                'importance': 'sum',
                'importance_normalized': 'sum',
                'sensor_type': 'first',
                'location': 'first',
                'axis': 'first'
            }).reset_index()

            # Normalize sensor importance
            if sensor_agg['importance'].max() > 0:
                sensor_agg['sensor_importance_normalized'] = sensor_agg['importance'] / sensor_agg['importance'].max()
            else:
                sensor_agg['sensor_importance_normalized'] = 0

            # Store results
            results[task] = {
                'feature_importance': importance_df,
                'feature_sensor_mapping': sensor_importance_df,
                'sensor_importance': sensor_agg
            }

            print(f"Processed {task}: Found {len(sensor_agg)} unique sensors")

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    return results


def visualize_sensor_importance(results, output_dir=None):

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sensor_type_colors = {
        'EMG': 'skyblue',
        'IMU': 'salmon',
        'Unknown': 'gray'
    }

    for task, data in results.items():
        sensor_importance = data['sensor_importance']

        if len(sensor_importance) > 0:
            top_sensors = sensor_importance.sort_values('importance', ascending=False).head(20)

            plt.figure(figsize=(12, 8))

            bar_colors = [sensor_type_colors.get(sensor_type, 'gray') for sensor_type in top_sensors['sensor_type']]

            sns.barplot(x='sensor_importance_normalized', y='sensor_id',
                        data=top_sensors, palette=bar_colors)

            plt.title(f'Top 20 Most Important Sensors for {task}')
            plt.xlabel('Normalized Importance')
            plt.ylabel('Sensor ID')
            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, f'sensor_importance_{task}.png'), dpi=300)
                plt.close()
            else:
                plt.show()

    # Create comparison of sensor importance across tasks
    # Find sensors that appear in multiple tasks
    common_sensors = set()
    for task, data in results.items():
        sensor_importance = data['sensor_importance']
        top_sensors = sensor_importance.sort_values('importance', ascending=False).head(10)['sensor_id'].tolist()
        common_sensors.update(top_sensors)

    comparison_data = []
    for sensor in common_sensors:
        sensor_data = {'sensor_id': sensor}

        for task, data in results.items():
            sensor_importance = data['sensor_importance']
            if sensor in sensor_importance['sensor_id'].values:
                importance = \
                sensor_importance[sensor_importance['sensor_id'] == sensor]['sensor_importance_normalized'].values[0]
                sensor_data[task] = importance
            else:
                sensor_data[task] = 0

        comparison_data.append(sensor_data)

    comparison_df = pd.DataFrame(comparison_data)

    task_columns = [col for col in comparison_df.columns if col != 'sensor_id']
    comparison_df['avg_importance'] = comparison_df[task_columns].mean(axis=1)
    comparison_df = comparison_df.sort_values('avg_importance', ascending=False).head(15)

    plt.figure(figsize=(15, 10))
    heatmap_data = comparison_df.set_index('sensor_id')[task_columns]
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', linewidths=.5)
    plt.title('Top Sensor Importance Comparison Across Tasks')
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'sensor_importance_comparison.png'), dpi=300)
        plt.close()
    else:
        plt.show()


    optimal_subsets = {}
    for task, data in results.items():
        sensor_importance = data['sensor_importance'].sort_values('importance', ascending=False).copy()

        sensor_importance['cumulative_importance'] = sensor_importance['sensor_importance_normalized'].cumsum()

        threshold_90 = sensor_importance[sensor_importance['cumulative_importance'] >= 0.9]

        if len(threshold_90) > 0:
            min_sensors_90 = threshold_90.iloc[0]['sensor_id']
            count_90 = threshold_90.index[0] + 1
        else:
            min_sensors_90 = None
            count_90 = len(sensor_importance)

        # Store optimal subsets
        optimal_subsets[task] = {
            'sensor_count_90': count_90,
            'sensors_90': sensor_importance.head(count_90)['sensor_id'].tolist()
        }

    return comparison_df, optimal_subsets


def find_optimal_sensor_subset(results):

    all_sensors = set()
    sensor_importance_by_task = {}

    for task, data in results.items():
        sensor_importance = data['sensor_importance']
        sensor_importance_by_task[task] = {}

        for _, row in sensor_importance.iterrows():
            sensor_id = row['sensor_id']
            importance = row['sensor_importance_normalized']

            all_sensors.add(sensor_id)
            sensor_importance_by_task[task][sensor_id] = importance

    # Calculate overall importance of each sensor across tasks
    sensor_scores = {}
    for sensor in all_sensors:
        # Average importance across tasks (using 0 if sensor not present in a task)
        scores = [sensor_importance_by_task[task].get(sensor, 0) for task in sensor_importance_by_task]
        sensor_scores[sensor] = sum(scores) / len(scores)

    # Sort sensors by overall importance
    sorted_sensors = sorted(sensor_scores.items(), key=lambda x: x[1], reverse=True)

    # Define different subsets for different coverage levels
    sensor_subsets = {
        'top_5': [s[0] for s in sorted_sensors[:5]],
        'top_10': [s[0] for s in sorted_sensors[:10]],
        'top_20': [s[0] for s in sorted_sensors[:20]]
    }

    # Calculate how much importance is covered by each subset for each task
    coverage = {}
    for subset_name, sensors in sensor_subsets.items():
        coverage[subset_name] = {}

        for task, importances in sensor_importance_by_task.items():
            # Sum importance of selected sensors for this task
            covered_importance = sum(importances.get(sensor, 0) for sensor in sensors)
            # Get total importance for this task
            total_importance = sum(importances.values())

            if total_importance > 0:
                coverage_percent = covered_importance / total_importance * 100
            else:
                coverage_percent = 0

            coverage[subset_name][task] = coverage_percent

    # Find the minimal subset that covers at least 80% importance for each task
    sensor_list = [s[0] for s in sorted_sensors]
    min_subset = []

    for task, importances in sensor_importance_by_task.items():
        # Sort sensors by importance for this specific task
        task_sensors = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        # Calculate how many sensors needed for 80% coverage
        total = sum(importances.values())
        cumulative = 0
        task_min_subset = []

        for sensor, importance in task_sensors:
            task_min_subset.append(sensor)
            cumulative += importance

            if total > 0 and cumulative / total >= 0.8:
                break

        # Add these sensors to the overall minimum subset
        min_subset.extend([s for s in task_min_subset if s not in min_subset])

    # Calculate coverage of the minimal subset
    min_subset_coverage = {}
    for task, importances in sensor_importance_by_task.items():
        covered_importance = sum(importances.get(sensor, 0) for sensor in min_subset)
        total_importance = sum(importances.values())

        if total_importance > 0:
            coverage_percent = covered_importance / total_importance * 100
        else:
            coverage_percent = 0

        min_subset_coverage[task] = coverage_percent

    return {
        'sensor_subsets': sensor_subsets,
        'coverage': coverage,
        'minimal_subset': min_subset,
        'minimal_subset_coverage': min_subset_coverage
    }


def generate_report(results, comparison_df, optimal_subsets, output_dir=None):

    report = "# Sensor Optimization Analysis Report\n\n"

    # Summary of analyzed tasks
    report += "## Tasks Analyzed\n\n"
    for task in results.keys():
        report += f"- {task}\n"

    report += "\n## Top 5 Sensors by Task\n\n"

    # Report top sensors for each task
    for task, data in results.items():
        report += f"### {task}\n\n"

        sensor_importance = data['sensor_importance']
        if len(sensor_importance) > 0:
            top_5 = sensor_importance.sort_values('importance', ascending=False).head(5)

            report += "| Sensor ID | Sensor Type | Location | Axis | Normalized Importance |\n"
            report += "|-----------|------------|----------|------|----------------------|\n"

            for _, row in top_5.iterrows():
                report += f"| {row['sensor_id']} | {row['sensor_type']} | {row['location']} | {row['axis']} | {row['sensor_importance_normalized']:.4f} |\n"

        report += "\n"

    # Report on optimal sensor subsets
    report += "## Optimal Sensor Subsets\n\n"

    report += "### Minimum Sensors Required for 90% Importance Coverage\n\n"

    report += "| Task | Number of Sensors | Sensor List |\n"
    report += "|------|-----------------|------------|\n"

    for task, subset_info in optimal_subsets.items():
        sensors_list = ", ".join(subset_info['sensors_90'][:5])
        if len(subset_info['sensors_90']) > 5:
            sensors_list += f", ... ({len(subset_info['sensors_90']) - 5} more)"

        report += f"| {task} | {subset_info['sensor_count_90']} | {sensors_list} |\n"

    # Save the report
    if output_dir:
        report_path = os.path.join(output_dir, "sensor_optimization_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_path}")

    return report


def run_sensor_optimization_analysis(feature_mapping_path, feature_importance_dir, output_dir=None):

    print("Starting sensor optimization analysis...")

    # Step 1: Analyze sensor importance
    results = analyze_sensor_importance(feature_mapping_path, feature_importance_dir)

    if not results:
        print("Analysis failed - no results to process.")
        return

    # Post-process to fix sensor ID issues
    print("Post-processing sensor IDs...")
    for task, data in results.items():
        # Fix sensor IDs in the sensor importance dataframe
        if 'sensor_importance' in data:
            sensor_df = data['sensor_importance']

            # Create new corrected sensor IDs
            corrected_ids = []
            for _, row in sensor_df.iterrows():
                sensor_type = row['sensor_type']
                location = row['location']

                # For IMU sensors, remove the axis from sensor_id
                if sensor_type == 'IMU':
                    corrected_id = f"{sensor_type}_{location}"
                else:
                    # For EMG sensors, ensure location is included
                    corrected_id = f"{sensor_type}_{location}"

                corrected_ids.append(corrected_id)

            # Add corrected IDs as a new column
            sensor_df['corrected_sensor_id'] = corrected_ids

            # Replace the original sensor_id with corrected version
            sensor_df['original_sensor_id'] = sensor_df['sensor_id']  # Save original for reference
            sensor_df['sensor_id'] = sensor_df['corrected_sensor_id']

            # Drop the temporary column
            sensor_df.drop('corrected_sensor_id', axis=1, inplace=True)

        # Similarly fix sensor IDs in the feature-sensor mapping
        if 'feature_sensor_mapping' in data:
            mapping_df = data['feature_sensor_mapping']

            # Create new corrected sensor IDs
            corrected_ids = []
            for _, row in mapping_df.iterrows():
                sensor_type = row['sensor_type']
                location = row['location']

                # For IMU sensors, remove the axis from sensor_id
                if sensor_type == 'IMU':
                    corrected_id = f"{sensor_type}_{location}"
                else:
                    # For EMG sensors, ensure location is included
                    corrected_id = f"{sensor_type}_{location}"

                corrected_ids.append(corrected_id)

            # Add corrected IDs as a new column
            mapping_df['corrected_sensor_id'] = corrected_ids

            # Replace the original sensor_id with corrected version
            mapping_df['original_sensor_id'] = mapping_df['sensor_id']  # Save original for reference
            mapping_df['sensor_id'] = mapping_df['corrected_sensor_id']

            # Drop the temporary column
            mapping_df.drop('corrected_sensor_id', axis=1, inplace=True)

    # Re-aggregate importance by the corrected sensor IDs
    print("Re-aggregating importance by corrected sensor IDs...")
    for task, data in results.items():
        if 'feature_sensor_mapping' in data:
            # Get the feature-sensor mapping with corrected IDs
            feature_sensor_df = data['feature_sensor_mapping']

            # Re-aggregate by the corrected sensor IDs
            sensor_agg = feature_sensor_df.groupby('sensor_id').agg({
                'importance': 'sum',
                'importance_normalized': 'sum',
                'sensor_type': 'first',
                'location': 'first',
                'axis': 'first'
            }).reset_index()

            # Normalize sensor importance
            if sensor_agg['importance'].max() > 0:
                sensor_agg['sensor_importance_normalized'] = sensor_agg['importance'] / sensor_agg['importance'].max()
            else:
                sensor_agg['sensor_importance_normalized'] = 0

            # Replace the sensor importance dataframe
            data['sensor_importance'] = sensor_agg

    # Step 2: Visualize sensor importance
    print("Creating visualizations...")
    comparison_df, optimal_subsets = visualize_sensor_importance(results, output_dir)

    # Step 3: Find optimal sensor subset
    print("Finding optimal sensor subsets...")
    sensor_subset_info = find_optimal_sensor_subset(results)

    # Step 4: Generate report
    print("Generating final report...")
    report = generate_report(results, comparison_df, optimal_subsets, output_dir)

    print("Analysis complete!")

    return {
        'results': results,
        'comparison': comparison_df,
        'optimal_subsets': optimal_subsets,
        'sensor_subset_info': sensor_subset_info,
        'report': report
    }


def extract_sensor_from_feature(feature_name):

    # Check for EMG features (muscle names)
    emg_muscles = ['trap', 'del', 'lat']
    for muscle in emg_muscles:
        if muscle in feature_name:
            return f"EMG_{muscle}"

    # Check for IMU features - these will have location names
    imu_locations = ['Shoulder', 'Upperarm', 'Forearm', 'Hand', 'Torso']

    for location in imu_locations:
        if location in feature_name:
            return f"IMU_{location}"

    # If we get here and still haven't identified a sensor, look for patterns
    if 'EMG' in feature_name:
        for muscle in emg_muscles:
            if muscle in feature_name:
                return f"EMG_{muscle}"

    # Last attempt to find any location
    if any(loc in feature_name for loc in imu_locations):
        for loc in imu_locations:
            if loc in feature_name:
                return f"IMU_{loc}"

    # If truly no match found, mark as unknown rather than returning feature name
    return "Unknown"

def aggregate_sensor_importance(importance_dir, output_path=None, threshold=0.1):

    all_task_importances = {}

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(importance_dir, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {importance_dir}")

    # Process each task file
    for file_path in csv_files:
        # Extract task name from filename
        file_name = os.path.basename(file_path)
        # Handles format like "feature_importance_TaskName.csv"
        if file_name.startswith("feature_importance_"):
            task_name = file_name.replace("feature_importance_", "").replace(".csv", "")
        else:
            # Fallback for other naming conventions
            task_name = file_name.split(".csv")[0]

        print(f"Processing task: {task_name} from file: {file_name}")

        # Read importance file
        task_data = pd.read_csv(file_path)

        # Check for the expected columns based on your file format
        if 'Feature' in task_data.columns and 'Importance' in task_data.columns:
            # Extract sensor information from feature names
            task_data['Sensor_ID'] = task_data['Feature'].apply(extract_sensor_from_feature)

            # Group by sensor and aggregate importance
            sensor_importance = task_data.groupby('Sensor_ID')['Importance'].sum().reset_index()

            # Normalize importance scores
            max_importance = sensor_importance['Importance'].max()
            if max_importance > 0:
                sensor_importance['Importance'] = sensor_importance['Importance'] / max_importance

            # Store importance scores for this task
            all_task_importances[task_name] = sensor_importance
        else:
            print(f"Warning: {file_path} does not have the expected column format. Skipping.")

    # Get unique list of all sensors across all tasks
    all_sensors = set()
    for task_data in all_task_importances.values():
        all_sensors.update(task_data['Sensor_ID'].unique())

    # Create empty dataframe for aggregated results
    result = pd.DataFrame({
        'Sensor': list(all_sensors),
        'Avg_Importance': 0.0,
        'Max_Importance': 0.0,
        'Important_Tasks': ''
    })

    # Calculate average importance and find tasks where sensor is important
    for sensor in all_sensors:
        importances = []
        important_tasks = []

        for task_name, task_data in all_task_importances.items():
            # Get importance for this sensor in this task (if exists)
            sensor_data = task_data[task_data['Sensor_ID'] == sensor]

            if not sensor_data.empty:
                importance = sensor_data['Importance'].iloc[0]
                importances.append(importance)

                # Check if importance exceeds threshold
                if importance >= threshold:
                    important_tasks.append(task_name)

        # Update result dataframe
        if importances:
            idx = result[result['Sensor'] == sensor].index[0]
            result.at[idx, 'Avg_Importance'] = np.mean(importances)
            result.at[idx, 'Max_Importance'] = np.max(importances)
            result.at[idx, 'Important_Tasks'] = ', '.join(important_tasks)

    # Sort by average importance in descending order
    result = result.sort_values('Avg_Importance', ascending=False).reset_index(drop=True)

    # Save results if output path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save to CSV
        result.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    return result