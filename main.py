from model_trainer import train_task
import os
from borg_labeling import load_data, assign_rpe_to_repetitions, restructure_borg_data, process_all_tasks
from add_borg_cubic_to_features import add_cubic_borg_to_features_for_all_tasks

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


# ------------------------------------------------------------------------------------

# Patricia Dataset Individual Tasks

    patricia_tasks = ['StSh25', 'StSh45', 'StEl25', 'StEl45', 'DySh25', 'DySh45', 'DyEl25', 'DyEl25']

    IMU_Sensors_Patricia = ['Forearm', 'Torso', 'Hand', 'Shoulder', 'Upperarm']
    EMG_Sensors_Patricia = ["Del", "Trap", "Lat"]

    patricia_features_directory = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features/Labeled"

    

    # for task in patricia_tasks:
    #     patricia_labeled_task_features = f"{patricia_features_directory}/{task}/labeled_features_{task}.csv"
    #     print(patricia_labeled_task_features)
    #
    #     patricia_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features/Labeled/"
    #     patricia_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Models"
    #
    #     # Train models for all tasks
    #     all_task_metrics = train_task(patricia_labeled_dir, patricia_models_results_dir, task)
    #
    #     print(f"\n[SUCCESS] Model training complete for {task}")
