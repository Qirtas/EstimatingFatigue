from model_trainer import train_task
import os
from borg_labeling import load_data, assign_rpe_to_repetitions, restructure_borg_data, process_all_tasks
from add_borg_cubic_to_features import add_cubic_borg_to_features_for_all_tasks
from PatriciaFeaturesFiltering import filter_emg_features
from MarcoCombinedTasks import merge_marco_tasks
from PatriciaCombinedTasks import merge_patricia_tasks

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

    # Sensor optimisation for both datasets combined


