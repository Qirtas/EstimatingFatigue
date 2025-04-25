import pandas as pd
import os
import glob


def add_borg_cubic_to_feature_file(task):
    """
    Add the Borg_Cubic column to the feature file for a specific task.

    Args:
        task: Task name (e.g., '55External')

    Returns:
        True if successful, False otherwise
    """
    # Define file paths
    repetition_borg_file = f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Borg/RepetitionTimes/{task}/repetition_times_with_borg.csv"
    features_dir = f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Features/Extracted/{task}/Labeled/"

    # Check if the repetition Borg file exists
    if not os.path.exists(repetition_borg_file):
        print(f"Error: Repetition Borg file not found for task {task}.")
        return False

    # Find all feature files in the directory
    feature_files = glob.glob(os.path.join(features_dir, "*.csv"))

    if not feature_files:
        print(f"Error: No feature files found in {features_dir}")
        return False

    # Load repetition Borg data
    try:
        borg_df = pd.read_csv(repetition_borg_file)
    except Exception as e:
        print(f"Error loading repetition Borg file for task {task}: {e}")
        return False

    # Create a mapping of (Subject, Repetition) -> Borg_RPE
    borg_mapping = {}
    for _, row in borg_df.iterrows():
        if pd.notna(row['Borg_RPE']):
            borg_mapping[(row['Subject'], row['Repetition'])] = row['Borg_RPE']

    # Process each feature file
    for feature_file in feature_files:
        print(f"Processing feature file: {os.path.basename(feature_file)}")

        try:
            # Load feature data
            feature_df = pd.read_csv(feature_file)

            # Add new Borg_Cubic column
            feature_df['Borg_Cubic'] = float('nan')

            # Assign Borg_Cubic values based on Subject and Repetition
            for idx, row in feature_df.iterrows():
                key = (row['Subject'], row['Repetition'])
                if key in borg_mapping:
                    feature_df.loc[idx, 'Borg_Cubic'] = borg_mapping[key]

            # Save updated feature file
            feature_df.to_csv(feature_file, index=False)
            print(f"Successfully added Borg_Cubic column to {os.path.basename(feature_file)}")

        except Exception as e:
            print(f"Error processing feature file {os.path.basename(feature_file)}: {e}")

    return True


def add_cubic_borg_to_features_for_all_tasks():
    """
    Process all tasks and add Borg_Cubic column to their feature files.
    """
    # Define tasks
    marco_tasks = ['35Internal', '35External', '45Internal', '45External', '55Internal', '55External']

    # Process each task
    for task in marco_tasks:
        print(f"\n{'=' * 40}")
        print(f"Processing task: {task}")
        print(f"{'=' * 40}\n")

        success = add_borg_cubic_to_feature_file(task)

        if success:
            print(f"Successfully processed task {task}")
        else:
            print(f"Failed to process task {task}")
