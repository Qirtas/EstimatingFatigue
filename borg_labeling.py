import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os


def load_data(borg_file, repetition_times_file):
    borg_df = pd.read_csv(borg_file)
    repetition_df = pd.read_csv(repetition_times_file)

    return borg_df, repetition_df


def restructure_borg_data(borg_df):

    restructured_data = []

    current_subject = None

    for idx, row in borg_df.iterrows():
        if not pd.isna(row['subject']):
            current_subject = row['subject']

        if current_subject is not None:
            new_row = row.copy()
            new_row['subject'] = current_subject
            restructured_data.append(new_row)

    return pd.DataFrame(restructured_data)


def get_interpolated_borg_value(times, rpe_values, midpoint, kind='cubic'):

    try:
        times = np.array(times)
        rpe_values = np.array(rpe_values)

        sorted_indices = np.argsort(times)
        times_sorted = times[sorted_indices]
        rpe_values_sorted = rpe_values[sorted_indices]

        f = interp1d(
            times_sorted,
            rpe_values_sorted,
            kind=kind,
            bounds_error=False,
            fill_value=(rpe_values_sorted[0], rpe_values_sorted[-1])
        )

        result = float(f(midpoint))
        return result
    except Exception as e:
        print(f"Error interpolating at midpoint {midpoint}: {e}")
        return None


def assign_rpe_to_repetitions(repetition_df, borg_df, task_name="task6_55e"):

    restructured_borg = restructure_borg_data(borg_df)

    result_df = repetition_df.copy()

    result_df['Borg_RPE'] = np.nan

    for subject, subject_group in repetition_df.groupby('Subject'):
        subject_id_str = f"subject_{subject}"

        subject_data = restructured_borg[restructured_borg['subject'] == subject_id_str]

        if len(subject_data) == 0:
            print(f"No Borg data found for subject {subject}")
            continue

        task_data = subject_data[subject_data['task_order'] == task_name]

        if len(task_data) == 0:
            print(f"No data found for task {task_name} for subject {subject}")
            continue

        row = task_data.iloc[0]
        print(f"Found task data for {subject_id_str}, task: {task_name}")

        time_columns = [col for col in restructured_borg.columns if
                        ('sec' in col and col != 'length_of_trial_(sec)') or col == 'before_task']

        times = []
        rpe_values = []

        for col in time_columns:
            # Skip if value is NaN
            if pd.isna(row[col]):
                continue

            if col == 'before_task':
                time_val = 0
            else:
                time_val = int(col.split('_')[0])

            if not pd.isna(row[col]) and row[col] != '':
                try:
                    rpe_val = float(row[col])
                    times.append(time_val)
                    rpe_values.append(rpe_val)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert '{row[col]}' to float. Skipping.")

        interp_kind = 'cubic'
        if len(times) < 4:
            print(f"Warning: Only found {len(times)} valid time points for subject {subject}, task {task_name}.")
            if len(times) >= 2:
                interp_kind = 'linear'
            else:
                print(f"Error: Not enough valid data points for interpolation.")
                continue

        for idx in subject_group.index:
            midpoint = subject_group.loc[idx, 'Midpoint']

            # Apply interpolation to get RPE at midpoint
            interpolated_rpe = get_interpolated_borg_value(times, rpe_values, midpoint, kind=interp_kind)

            if interpolated_rpe is not None:
                # Store the interpolated RPE
                result_df.loc[idx, 'Borg_RPE'] = round(interpolated_rpe, 1)
                print(
                    f"Assigned Borg RPE {round(interpolated_rpe, 1)} to repetition at midpoint {midpoint} for subject {subject}")

    return result_df


def get_task_name_mapping(task):

    # Mapping from task names to task_order values in Borg data
    task_mapping = {
        '35Internal': 'task1_35i',
        '45Internal': 'task2_45i',
        '55Internal': 'task3_55i',
        '35External': 'task4_35e',
        '45External': 'task5_45e',
        '55External': 'task6_55e'
    }

    return task_mapping.get(task)


def process_all_tasks():
    marco_tasks = ['35Internal', '35External', '45Internal', '45External', '55Internal', '55External']

    marco_borg_raw_file = "/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Borg/borg_data.csv"

    for task in marco_tasks:
        print(f"\n{'=' * 40}")
        print(f"Processing task: {task}")
        print(f"{'=' * 40}\n")

        task_order_name = get_task_name_mapping(task)

        if task_order_name is None:
            print(f"Error: No mapping found for task {task}. Skipping.")
            continue

        repetition_times_file = f"/Volumes/Visentin_Re/EstimatingFatigue/MarcoDataset/Borg/RepetitionTimes/{task}/repetition_times_all_subjects.csv"

        output_dir = os.path.dirname(repetition_times_file)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(repetition_times_file):
            print(f"Warning: Repetition times file not found for task {task}. Skipping.")
            continue

        output_file = os.path.join(output_dir, "repetition_times_with_borg.csv")

        print(f"Loading data for task {task}...")
        borg_df, repetition_df = load_data(marco_borg_raw_file, repetition_times_file)

        print(f"Assigning Borg RPE values to repetitions for task {task}...")
        result_df = assign_rpe_to_repetitions(repetition_df, borg_df, task_name=task_order_name)

        assigned_count = result_df['Borg_RPE'].notna().sum()
        total_count = len(result_df)
        print(
            f"Successfully assigned Borg RPE values to {assigned_count} out of {total_count} repetitions for task {task}.")

        print(f"Saving results to {output_file}...")
        result_df.to_csv(output_file, index=False)
        print(f"Done processing task {task}!")