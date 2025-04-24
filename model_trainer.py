import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import GroupKFold, cross_val_predict, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_and_evaluate(csv_path, task_name, output_dir):
    """
    Train and evaluate models using all features, then use the top features
    from Random Forest to retrain and evaluate models again.

    Parameters:
    - csv_path (str): Path to the feature file.
    - task_name (str): Name of the task (e.g., 'StSh25').
    - output_dir (str): Directory to save results.
    """
    # print(f"\n=== Loading Data for {task_name} ===\n")
    # try:
    #     df = pd.read_csv(csv_path)
    #     print(f"Data loaded successfully from '{csv_path}'.")
    #     print(f"Shape: {df.shape}, Features: {df.shape[1]}")
    # except FileNotFoundError:
    #     print(f"Error: File '{csv_path}' not found.")
    #     return
    # except Exception as e:
    #     print(f"Error loading data: {e}")
    #     return
    #
    # # Check if RPE column exists
    # if 'RPE' not in df.columns:
    #     print(f"Error: 'RPE' column not found in the dataset. Available columns: {df.columns.tolist()}")
    #     return
    #
    # # Drop rows with missing RPE values
    # df = df.dropna(subset=['RPE'])
    # print(f"After dropping rows with missing RPE: {df.shape}")
    #
    # # Define target and features
    # y = df['RPE']
    #
    # # Identify columns to drop (metadata columns)
    # cols_to_drop = ['Subject', 'Repetition', 'RPE', 'Task']

    print(f"\n=== Loading Data for {task_name} ===\n")
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully from '{csv_path}'.")
        print(f"Shape: {df.shape}, Features: {df.shape[1]}")
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check if RPE or BORG column exists
    target_column = None
    if 'RPE' in df.columns:
        target_column = 'RPE'
        print("Using 'RPE' as target column.")
    elif 'Borg' in df.columns:
        target_column = 'Borg'
        print("Using 'BORG' as target column. This appears to be the equivalent of RPE in this dataset.")
    else:
        print(f"Error: Neither 'RPE' nor 'BORG' column found in the dataset. Available columns: {df.columns.tolist()}")
        return

    # Drop rows with missing target values
    df = df.dropna(subset=[target_column])
    print(f"After dropping rows with missing {target_column}: {df.shape}")

    # Define target and features
    y = df[target_column]

    # Identify columns to drop (metadata columns)
    cols_to_drop = ['Subject', 'Repetition', 'Task', target_column]
    # Add the target column to cols_to_drop

    # Add any other non-feature columns that might be in your dataset
    for col in df.columns:
        if 'Unnamed' in col:
            cols_to_drop.append(col)

    # Drop metadata columns to get features
    X = df.drop(cols_to_drop, axis=1, errors='ignore')
    groups = df['Subject']

    print(f"Target variable (RPE) statistics:")
    print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    print(f"Number of features: {X.shape[1]}")

    # Define models to train
    models = {
        'XGBoost Regressor': XGBRegressor(random_state=42, n_jobs=-1)
        #'Support Vector Regressor': SVR()
        #'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1)
    }

    # Set up cross-validation
    n_splits = len(df['Subject'].unique())
    print(f"Number of GroupKFold Splits: {n_splits}")
    group_kfold = GroupKFold(n_splits=n_splits)

    # Create results directory
    results_dir = os.path.join(output_dir, task_name, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Identify top features using Random Forest
    print(f"\n=== Identifying Top Features for {task_name} ===\n")
    feature_imp_path = os.path.join(results_dir, f'feature_importance_{task_name}.csv')
    top_features = get_top_features(X, y, feature_imp_path, 150)

    # Reduce dataset to top features
    X_top = X[top_features]
    print(f"Dataset reduced to top {len(top_features)} features")

    # Step 2: Train and evaluate using top features
    print(f"\n=== Training and Evaluating Models for {task_name} ===\n")
    performance_metrics = train_models(models, X_top, y, groups, group_kfold, task_name, results_dir)

    # Save performance metrics to CSV
    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
    metrics_path = os.path.join(results_dir, f'performance_metrics_{task_name}.csv')
    metrics_df.to_csv(metrics_path)
    print(f"Performance metrics saved to: {metrics_path}")

    return performance_metrics



def train_models(models, X, y, groups, group_kfold, task_name, results_dir):
    """
    Train and evaluate models using GroupKFold cross-validation.

    Parameters:
    - models (dict): Dictionary of model names and instances.
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.
    - groups (pd.Series): Group labels for GroupKFold.
    - group_kfold (GroupKFold): Cross-validation strategy.
    - task_name (str): Name of the task.
    - results_dir (str): Directory to save results.

    Returns:
    - performance_metrics (dict): Performance metrics for each model.
    """
    performance_metrics = {}

    for model_name, model in models.items():
        print(f"\n--- Evaluating {model_name} for {task_name} ---")

        if model_name == 'XGBoost Regressor':
            # Perform hyperparameter tuning for XGBoost
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [3, 5, 7],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__min_child_weight': [1, 3, 5],
                'regressor__subsample': [0.8, 0.9, 1.0]
            }

            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=20,
                cv=group_kfold,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )

            print("[INFO] Tuning XGBoost hyperparameters...")
            random_search.fit(X, y, groups=groups)

            best_model = random_search.best_estimator_
            print(f"Best Parameters for XGBoost: {random_search.best_params_}")

        else:
            # For models without hyperparameter tuning, use the default model
            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])

            best_model = pipeline

        # Perform cross-validation predictions
        print(f"[INFO] Performing cross-validation for {model_name}...")
        y_pred = cross_val_predict(
            best_model,
            X,
            y,
            cv=group_kfold,
            groups=groups,
            n_jobs=-1
        )

        # Compute performance metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mape = (abs((y - y_pred) / y).mean()) * 100

        performance_metrics[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

        # Save actual vs predicted plot
        plot_path = save_actual_vs_predicted_plot(y, y_pred, model_name, task_name, results_dir)

        # Print performance metrics
        print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
        print(f"Plot saved to: {plot_path}")

    return performance_metrics



def save_actual_vs_predicted_plot(y, y_pred, model_name, task_name, save_dir):
    """
    Create and save a scatter plot of actual vs. predicted values.

    Parameters:
    - y (pd.Series): Actual values
    - y_pred (np.array): Predicted values
    - model_name (str): Name of the model
    - task_name (str): Name of the task
    - save_dir (str): Directory to save the plot

    Returns:
    - str: Path to the saved plot
    """

    plt.figure(figsize=(10, 8))

    # Create scatter plot
    sns.scatterplot(x=y, y=y_pred, alpha=0.6)

    # Add perfect prediction line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    # Add regression line
    # sns.regplot(x=y, y=y_pred, scatter=False, color='blue', label='Regression Line')

    # Add labels and title
    plt.xlabel('Actual RPE')
    plt.ylabel('Predicted RPE')
    plt.title(f'{model_name}: Actual vs. Predicted RPE - {task_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add performance metrics to the plot
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    plt.annotate(f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Create filename with model name and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_{model_name.replace(' ', '_')}_actual_vs_predicted_{timestamp}.png"
    full_path = os.path.join(save_dir, filename)

    # Save the plot
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close()

    return full_path


def get_top_features(X, y, feature_imp_path, n_features=150):
    """
    Get top n features using Random Forest feature importance

    Parameters:
    ----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    feature_imp_path : str
        Path to save feature importance plot
    n_features : int
        Number of top features to select

    Returns:
    -------
    list
        List of top feature names
    """
    print("[INFO] Identifying and removing problematic features...")

    problematic_cols = []
    for col in X.columns:
        try:
            # First check if the column is numeric
            if pd.api.types.is_numeric_dtype(X[col]):
                # Then check for infinities and extremely large values
                if np.any(np.isinf(X[col])) or np.any(np.abs(X[col]) > 1e15):
                    problematic_cols.append(col)
                    print(f"[WARN] Removing problematic feature: {col} (infinity or extreme value)")
            else:
                # If column is not numeric, mark it as problematic
                problematic_cols.append(col)
                print(f"[WARN] Removing problematic feature: {col} (non-numeric data)")
        except Exception as e:
            # Catch any other errors and add the column to problematic list
            problematic_cols.append(col)
            print(f"[WARN] Removing problematic feature: {col} (error: {str(e)})")

    # Remove problematic columns
    if problematic_cols:
        X_clean = X.drop(columns=problematic_cols)
        print(f"[INFO] Removed {len(problematic_cols)} problematic features")
    else:
        X_clean = X
        print("[INFO] No problematic features found")

    # Check if we have any features left
    if X_clean.shape[1] == 0:
        print("[ERROR] No features left after removing problematic ones")
        return []

    print(f"[INFO] Fitting Random Forest model for feature importance with {X_clean.shape[1]} features...")

    # Create pipeline with preprocessing and model
    rf_pipeline = Pipeline([
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('scaler', RobustScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Fit the pipeline
    rf_pipeline.fit(X_clean, y)

    # Get feature importances
    feature_importances = rf_pipeline.named_steps['rf'].feature_importances_

    # Handle potential length mismatch between features and importances
    # This can happen if the VarianceThreshold step removes some features
    var_thresh = rf_pipeline.named_steps['var_thresh']
    # Get indices of features that were kept after variance thresholding
    if hasattr(var_thresh, 'get_support'):
        # Get mask of features that passed the variance threshold
        mask = var_thresh.get_support()
        # Filter the original feature names using this mask
        features_after_var_thresh = X_clean.columns[mask]

        print(
            f"[INFO] Features after variance thresholding: {len(features_after_var_thresh)} (from {len(X_clean.columns)})")

        # Now create DataFrame with the correct feature names
        feature_importance_df = pd.DataFrame({
            'Feature': features_after_var_thresh,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
    else:
        # Fallback if get_support() is not available (should not happen)
        print("[WARN] Could not determine features after variance thresholding, using a workaround")
        # Create a fresh DataFrame with just the top features we're confident about
        top_n = min(len(feature_importances), len(X_clean.columns))

        # Sort importances in descending order and get top_n
        sorted_idx = np.argsort(feature_importances)[::-1][:top_n]
        top_features = [X_clean.columns[i] for i in sorted_idx]
        top_importances = [feature_importances[i] for i in sorted_idx]

        feature_importance_df = pd.DataFrame({
            'Feature': top_features,
            'Importance': top_importances
        })

    # Plot feature importances
    # plt.figure(figsize=(12, 8))
    # sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(min(30, len(feature_importance_df))))
    # plt.title('Top 30 Feature Importances')
    # plt.tight_layout()
    #
    # # Create directory if it doesn't exist
    # os.makedirs(os.path.dirname(feature_imp_path), exist_ok=True)
    #
    # # Save plot
    # plt.savefig(feature_imp_path)
    # plt.close()

    print("\nTop 10 Features:")
    for i, (feature, importance) in enumerate(zip(feature_importance_df['Feature'].head(10),
                                                  feature_importance_df['Importance'].head(10))):
        print(f"{i + 1}. {feature}: {importance:.4f}")

    # Get top n features
    top_features = feature_importance_df.head(min(n_features, len(feature_importance_df)))['Feature'].tolist()
    feature_importance_df.head(n_features).to_csv(feature_imp_path, index=False)
    print(f"Top Features Saved at {len(feature_imp_path)}")

    print(f"[INFO] Selected {len(top_features)} top features")

    return top_features


def train_task(labeled_dir, output_dir, task, analyze_errors=True, combined_task=False, combined_file_path=None):
    """
    Train models for individual tasks or combined tasks.

    Parameters:
    - labeled_dir (str): Directory containing labeled feature files
    - output_dir (str): Directory to save results
    - task (str): Task name or identifier (e.g., 'DySh25' or 'CombinedDynamic')
    - analyze_errors (bool): Whether to perform error analysis
    - combined_task (bool): Whether this is a combined task training
    - combined_file_path (str): Path to the combined tasks feature file (required if combined_task is True)

    Returns:
    - dict: Dictionary containing task metrics
    """
    print(f"[INFO] Starting model training for {'combined' if combined_task else 'individual'} task: {task}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store performance metrics for all tasks
    all_task_metrics = {}

    if combined_task:
        # Handle combined task case
        if combined_file_path is None or not os.path.exists(combined_file_path):
            print(f"[ERROR] Combined task file path is required and must exist: {combined_file_path}")
            return {}

        print(f"\n{'=' * 50}")
        print(f"Processing combined task: {task}")
        print(f"{'=' * 50}")

        # Train and evaluate models for the combined task
        task_metrics = train_and_evaluate(
            csv_path=combined_file_path,
            task_name=task,
            output_dir=output_dir        )

        if task_metrics:
            all_task_metrics[task] = task_metrics
    elif task == "MarcoPatriciaMerged":
        print(f"Processing task: {task}")

        task_metrics = train_and_evaluate(
            csv_path=labeled_dir,
            task_name=task,
            output_dir=output_dir)

        if task_metrics:
            all_task_metrics[task] = task_metrics

    else:
        # Original code for individual tasks
        task_folders = [task]
        # Process each task
        for task_name in task_folders:
            print(f"\n{'=' * 50}")
            print(f"Processing task: {task_name}")
            print(f"{'=' * 50}")

            if task == "35Internal" or task == "35External" or task == "45Internal" or task == "45External" or task == "55Internal" or task == "55External":
                labeled_file_path = os.path.join(labeled_dir, f'labeled_features_{task_name}.csv')
                print(f"labeled_file_path: {labeled_file_path}")
            elif task == "AblationAllTasks":
                labeled_file_path = labeled_dir
                print(f"labeled_file_path: {labeled_file_path}")
            else:
                labeled_file_path = os.path.join(labeled_dir, task_name, f'labeled_features_{task_name}.csv')
                print(f"labeled_file_path: {labeled_file_path}")

            # Check if file exists
            if not os.path.exists(labeled_file_path):
                print(f"[WARN] Labeled features file not found for task {task_name}. Skipping.")
                continue

            # Train and evaluate models for this task
            task_metrics = train_and_evaluate(
                csv_path=labeled_file_path,
                task_name=task_name,
                output_dir=output_dir            )

            if task_metrics:
                all_task_metrics[task_name] = task_metrics

    # Create summary of all tasks
    #create_all_tasks_summary(all_task_metrics, output_dir)

    return all_task_metrics
