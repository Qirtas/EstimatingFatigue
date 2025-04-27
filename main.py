from model_trainer import train_task, train_baseline_model
import os
from borg_labeling import load_data, assign_rpe_to_repetitions, restructure_borg_data, process_all_tasks
from add_borg_cubic_to_features import add_cubic_borg_to_features_for_all_tasks
from PatriciaFeaturesFiltering import filter_emg_features
from MarcoCombinedTasks import merge_marco_tasks
from PatriciaCombinedTasks import merge_patricia_tasks
from SensorOptimisation.sensorImportanceAnalysis import ranking_sensor_importance
from SensorOptimisation.SensorsFiltering import make_sensor_subsets
from SensorOptimisation.AggregatedFeatureImportanceAnalysis import aggregate_sensor_importance
from FeaturesImportance.featureImpVisualisation import load_feature_importance_scores, aggregate_and_select_features, normalize_scores, create_heatmap

if __name__ == '__main__':

    # Setting up common sensors between both datasets

    marco_tasks = ['35Internal', '35External', '45Internal', '45External', '55Internal', '55External']

    IMU_Sensors_Marco = ['Shoulder', 'Forearm', 'Upperarm', 'Torso', 'Palm']
    EMG_Sensors_Marco = ['emg_deltoideus_anterior', 'emg_latissimus_dorsi', 'emg_trapezius_ascendens']

    marco_features_directory = "/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/Extracted"

    # BORG labeling

    marco_tasks = ['35Internal', '35External', '45Internal', '45External', '55Internal', '55External']

    # process_all_tasks()

    # add_cubic_borg_to_features_for_all_tasks()


    # Marco Dataset Individual Tasks Training

    # for task in marco_tasks:
    #     marco_labeled_task_features = f"{marco_features_directory}/{task}/labeled_features_{task}.csv"
    #     print(marco_labeled_task_features)
    #     marco_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/Extracted/{task}/Labeled/"
    #     marco_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Models"
    #
    #     # Train models for all tasks
    #     all_task_metrics = train_task(marco_labeled_dir, marco_models_results_dir, task)
    #
    #     print(f"\n[SUCCESS] Model training complete for {task}")


    # Marco Dataset Combined Tasks Training

    # merge_marco_tasks()

    # marco_tasks_combinations = ['AllInternal', 'AllExternal', 'AllCombined']
    # marco_features_directory =f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/{marco_tasks_combination}"


    # All Internal Tasks
    # all_internal_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/Combined/"
    # internal_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Models"
    #
    # all_task_metrics = train_task(all_internal_labeled_dir, internal_models_results_dir, 'AllInternal', target_column='Borg_Cubic')

    # All External Tasks
    # all_external_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/Combined/"
    # external_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Models"
    #
    # all_task_metrics = train_task(all_external_labeled_dir, external_models_results_dir, 'AllExternal',
    #                               target_column='Borg_Cubic')

    # All Combined

    # all_combined_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/Combined/"
    # all_combined_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Models"
    #
    # all_task_metrics = train_task(all_combined_labeled_dir, all_combined_models_results_dir, 'AllCombined',
    #                               target_column='Borg_Cubic')

# ------------------------------------------------------------------------------------

# Patricia Dataset Individual Tasks

    patricia_tasks = ['StSh25', 'StSh45', 'StEl25', 'StEl45', 'DySh25', 'DySh45', 'DyEl25', 'DyEl45']

    IMU_Sensors_Patricia = ['Forearm', 'Torso', 'Hand', 'Shoulder', 'Upperarm']
    EMG_Sensors_Patricia = ["Del", "Trap", "Lat"]
    emg_muscles = ["Del", "Trap", "Bicep", "Tricep", "Lat"]

    patricia_features_directory = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features/Labeled"

    # # List of excluded EMG sensors
    # excluded_sensors = ["Bicep", "Tricep"]
    #
    # print(f"Filtering out features containing any of these sensor names: {', '.join(excluded_sensors)}")
    #
    # # Filter the features
    # removed_counts = filter_emg_features(patricia_features_directory, excluded_sensors)
    #
    # # Print summary
    # print("\nSummary of removed features:")
    # for task, count in removed_counts.items():
    #     print(f"Task {task}: Removed {count} features")
    #
    # print("\nFeature filtering completed successfully!")

    # Train Individual Task Models

    # for task in patricia_tasks:
    #     patricia_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features/Labeled/"
    #     patricia_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Models"
    #
    #     # Train models for all tasks
    #     all_task_metrics = train_task(patricia_labeled_dir, patricia_models_results_dir, task)
    #
    #     print(f"\n[SUCCESS] Model training complete for {task}")


    # Patricia Dataset Combined Tasks Training
    # merge_patricia_tasks()

    # All Static Tasks
    # all_static_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features/"
    # static_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Models"
    #
    # all_static_task_metrics = train_task(all_static_labeled_dir, static_models_results_dir, 'AllStatic', target_column='RPE')

    # All Dynamic Tasks

    # all_dynamic_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features/"
    # dynamic_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Models"
    #
    # all_dynamic_task_metrics = train_task(all_dynamic_labeled_dir, dynamic_models_results_dir, 'AllDynamic',
    #                                      target_column='RPE')

    # All Combined Tasks

    # all_combined_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features/"
    # combined_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Models"
    #
    # all_combined_task_metrics = train_task(all_combined_labeled_dir, combined_models_results_dir, 'AllDynamic',
    #                                       target_column='RPE')


# ------------------------------------------------------------------------------------

    # Marco Patricia Merged

    # both_merged_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/"
    # both_merged_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Models"
    #
    # both_merged_task_metrics = train_task(both_merged_labeled_dir, both_merged_models_results_dir, 'BothMerged',
    #                                       target_column='RPE')


# ------------------------------------------------------------------------------------

    # Step 3: Sensor optimisation for both datasets combined

    # feature_sensor_mapping_file = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/FeatureSensorMapping/all_feature_mappings.csv"
    #
    # feature_importance_dir = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/FeatureImportanceScores"
    # sensor_importance_results = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/FeatureImportanceScores/SensorImportance/"
    #
    # results = ranking_sensor_importance(
    #     feature_mapping_path=feature_sensor_mapping_file,
    #     feature_importance_dir=feature_importance_dir,
    #     output_dir=sensor_importance_results
    # )


    # # Step 4: Aggregating Sensors Importance Across All Tasks
    #
    # feature_importance_dir = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/FeatureImportanceScores"
    # aggregated_sensor_imp_results = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/FeatureImpAnalysis/aggregated_sensor_importance.csv"
    #
    # # Process all files in the directory and save results
    # aggregated_importance = aggregate_sensor_importance(
    #     importance_dir=feature_importance_dir,
    #     output_path=aggregated_sensor_imp_results,
    #     threshold=0.1
    # )
    #
    # # Print results
    # print(aggregated_importance)


    # # Step 5: Features Filtering Based on Subset Sensors
    #
    # tasks_dir = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/TasksFeatureFiles"  # Directory with labeled_features_*.csv files
    # features_importance_file = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/FeatureImpAnalysis/aggregated_sensor_importance.csv"
    # sensor_subsets_output_dir = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/FilteredSensorSubsets"
    #
    # make_sensor_subsets(tasks_dir, features_importance_file, sensor_subsets_output_dir, max_subset_size=8)

    # Step 6: Models Training for each sensor subset

    # Sensor Subsets Model Training

    # base_labeled_dir = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/FilteredSensorSubsets/BothMerged"
    # sensor_subsets_models_result_dir = "/Volumes/Visentin_Re/EstimatingFatigue/Models/AblationResults"
    #
    # # Number of subsets to process
    # num_subsets = 8  # Change this if you have a different number of subsets
    #
    # # Loop through each subset
    # for subset_num in range(1, num_subsets + 1):
    #     print(f"\n{'=' * 50}")
    #     print(f"Processing Subset {subset_num}")
    #     print(f"{'=' * 50}")
    #
    #     # Construct path to the specific subset file
    #     labeled_dir = f"{base_labeled_dir}/subset_{subset_num}_sensors.csv"
    #
    #     # Create a unique output folder name for this subset
    #     output_name = f"AblationAllTasks_Subset{subset_num}"
    #
    #     # Train model for this subset
    #     print(f"Training model for subset {subset_num}...")
    #     subset_metrics = train_task(labeled_dir, sensor_subsets_models_result_dir, "AblationAllTasks", target_column='RPE')
    #
    #     # Optionally, print or process metrics for this subset
    #     print(f"Completed training for subset {subset_num}")
    #     print(f"Metrics: {subset_metrics}")
    #
    # print("\n[SUCCESS] Model training complete for all AllTasks subsets!")



# ------------------------------------------------------------------------------------

    # Baseline Model Training

    # both_merged_labeled_file = f"/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/labeled_features_BothMerged.csv"
    #
    # output_dir = "/Volumes/Visentin_Re/EstimatingFatigue/BothMerged/Ablation/BaselineModel"
    #
    # results = train_baseline_model(
    #     labeled_features_file=both_merged_labeled_file,
    #     output_dir=output_dir,
    #     target_column='RPE',
    #     subject_column='Subject'
    # )


# ------------------------------------------------------------------------------------

    # Heatmap Feature Importance Analysis

    features_imp_folder_path = "/Volumes/Visentin_Re/EstimatingFatigue/FeaturesImportance"
    feature_df = load_feature_importance_scores(features_imp_folder_path)

    top_features_df = aggregate_and_select_features(feature_df, top_n=10)

    normalized_df = normalize_scores(top_features_df)

    desired_order = ['AllStatic', 'AllDynamic', 'AllInternal', 'AllExternal', 'BothMerged', 'Aggregated_Importance']
    normalized_df = normalized_df.reindex(columns=desired_order)

    create_heatmap(normalized_df, title="")