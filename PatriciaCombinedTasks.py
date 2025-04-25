import pandas as pd
import os
import numpy as np


def merge_patricia_tasks():

    patricia_tasks = ['StSh25', 'StSh45', 'StEl25', 'StEl45', 'DySh25', 'DySh45', 'DyEl25', 'DyEl45']

    task_combinations = {
        'AllStatic': ['StSh25', 'StSh45', 'StEl25', 'StEl45'],
        'AllDynamic': ['DySh25', 'DySh45', 'DyEl25', 'DyEl45'],
        'AllCombined': patricia_tasks  # All tasks
    }

    input_base_path = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features/Labeled"

    output_base_path = "/Volumes/Visentin_Re/EstimatingFatigue/PatriciaDataset/Features"

    for combination_name, tasks_to_combine in task_combinations.items():
        print(f"\nProcessing combination: {combination_name}")

        dfs_to_merge = []

        for task in tasks_to_combine:
            task_file = os.path.join(input_base_path, task, f"labeled_features_{task}.csv")

            if not os.path.exists(task_file):
                print(f"Warning: File not found for task {task}: {task_file}")
                continue

            print(f"Loading features from {task}...")
            task_df = pd.read_csv(task_file)

            task_df['Task'] = task

            dfs_to_merge.append(task_df)
            print(f"  Added {len(task_df)} samples from {task}")

        if not dfs_to_merge:
            print(f"Error: No data found for combination {combination_name}")
            continue

        combined_df = pd.concat(dfs_to_merge, ignore_index=True)
        print(f"Combined dataset has {len(combined_df)} samples with {combined_df.shape[1]} features")

        output_dir = os.path.join(output_base_path, combination_name)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"labeled_features_{combination_name}.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"Saved combined features to {output_file}")

    print("\nTask merging complete!")
