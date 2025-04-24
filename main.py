from model_trainer import train_task


if __name__ == '__main__':
    import xgboost as xgb
    import sklearn

    print(f"XGBoost version: {xgb.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")

    # Setting up common sensors between both datasets

    marco_tasks = ['35Internal', '35External', '45Internal', '45External', '55Internal', '55External']

    IMU_Sensors_Marco = ['Shoulder', 'Forearm', 'Upperarm', 'Torso', 'Palm']
    EMG_Sensors_Marco = ['emg_deltoideus_anterior', 'emg_latissimus_dorsi', 'emg_trapezius_ascendens']

    marco_features_directory = "/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/Extracted"

    # for task in marco_tasks:
    #     marco_labeled_task_features = f"{marco_features_directory}/{task}/labeled_features_{task}.csv"
    #     print(marco_labeled_task_features)


# Marco Dataset Individual Tasks Training

    task = '55External'
    marco_labeled_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/Extracted/{task}/Labeled/"
    marco_models_results_dir = "/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Models"

    # Train models for all tasks
    all_task_metrics = train_task(marco_labeled_dir, marco_models_results_dir, task)

    print("\n[SUCCESS] Model training complete for all tasks!")



# ------------------------------------------------------------------------------------

# Patricia Dataset Individual Tasks