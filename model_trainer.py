import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
import datetime
from sklearn.model_selection import GroupKFold, cross_val_predict, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_and_evaluate(csv_path, task_name, output_dir, target_column):

    print(f"\n=== Loading Data for {task_name} and target_column is {target_column} ===\n")
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
    # target_column = None
    # if 'RPE' in df.columns:
    #     target_column = 'RPE'
    #     print("Using 'RPE' as target column.")
    print(f"Using {target_column} as target column.")

    df = df.dropna(subset=[target_column])
    print(f"After dropping rows with missing {target_column}: {df.shape}")

    y = df[target_column]

    cols_to_drop = ['Subject', 'Repetition', 'Task', target_column, 'Borg']

    for col in df.columns:
        if 'Unnamed' in col:
            cols_to_drop.append(col)

    X = df.drop(cols_to_drop, axis=1, errors='ignore')
    groups = df['Subject']

    print(f"Target variable (RPE) statistics:")
    print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    print(f"Number of features: {X.shape[1]}")

    models = {
        'XGBoost Regressor': XGBRegressor(random_state=42, n_jobs=-1)
        #'Support Vector Regressor': SVR()
        #'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1)
    }

    n_splits = len(df['Subject'].unique())
    print(f"Number of GroupKFold Splits: {n_splits}")
    group_kfold = GroupKFold(n_splits=n_splits)

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

    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
    metrics_path = os.path.join(results_dir, f'performance_metrics_{task_name}.csv')
    metrics_df.to_csv(metrics_path)
    print(f"Performance metrics saved to: {metrics_path}")

    return performance_metrics



def train_models(models, X, y, groups, group_kfold, task_name, results_dir):

    performance_metrics = {}

    for model_name, model in models.items():
        print(f"\n--- Evaluating {model_name} for {task_name} ---")

        if model_name == 'XGBoost Regressor':
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
            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])

            best_model = pipeline

        print(f"[INFO] Performing cross-validation for {model_name}...")
        y_pred = cross_val_predict(
            best_model,
            X,
            y,
            cv=group_kfold,
            groups=groups,
            n_jobs=-1
        )

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

        plot_path = save_actual_vs_predicted_plot(y, y_pred, model_name, task_name, results_dir)

        print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
        print(f"Plot saved to: {plot_path}")

    compute_and_save_shap(best_model, X, results_dir, task_name, max_display=20, beeswarm_display=10)

    return performance_metrics



def save_actual_vs_predicted_plot(y, y_pred, model_name, task_name, save_dir):


    plt.figure(figsize=(10, 8))

    sns.scatterplot(x=y, y=y_pred, alpha=0.6)

    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    plt.xlabel('Actual RPE')
    plt.ylabel('Predicted RPE')
    plt.title(f'{model_name}: Actual vs. Predicted RPE - {task_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    plt.annotate(f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_{model_name.replace(' ', '_')}_actual_vs_predicted_{timestamp}.png"
    full_path = os.path.join(save_dir, filename)

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close()

    return full_path


def get_top_features(X, y, feature_imp_path, n_features=150):

    print("[INFO] Identifying and removing problematic features...")

    problematic_cols = []
    for col in X.columns:
        try:
            if pd.api.types.is_numeric_dtype(X[col]):
                if np.any(np.isinf(X[col])) or np.any(np.abs(X[col]) > 1e15):
                    problematic_cols.append(col)
                    print(f"[WARN] Removing problematic feature: {col} (infinity or extreme value)")
            else:
                problematic_cols.append(col)
                print(f"[WARN] Removing problematic feature: {col} (non-numeric data)")
        except Exception as e:
            problematic_cols.append(col)
            print(f"[WARN] Removing problematic feature: {col} (error: {str(e)})")

    # Remove problematic columns
    if problematic_cols:
        X_clean = X.drop(columns=problematic_cols)
        print(f"[INFO] Removed {len(problematic_cols)} problematic features")
    else:
        X_clean = X
        print("[INFO] No problematic features found")

    if X_clean.shape[1] == 0:
        print("[ERROR] No features left after removing problematic ones")
        return []

    print(f"[INFO] Fitting Random Forest model for feature importance with {X_clean.shape[1]} features...")

    rf_pipeline = Pipeline([
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('scaler', RobustScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    rf_pipeline.fit(X_clean, y)

    feature_importances = rf_pipeline.named_steps['rf'].feature_importances_

    # Handle potential length mismatch between features and importances
    # This can happen if the VarianceThreshold step removes some features
    var_thresh = rf_pipeline.named_steps['var_thresh']
    # Get indices of features that were kept after variance thresholding
    if hasattr(var_thresh, 'get_support'):
        mask = var_thresh.get_support()
        features_after_var_thresh = X_clean.columns[mask]

        print(
            f"[INFO] Features after variance thresholding: {len(features_after_var_thresh)} (from {len(X_clean.columns)})")

        feature_importance_df = pd.DataFrame({
            'Feature': features_after_var_thresh,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
    else:
        print("[WARN] Could not determine features after variance thresholding, using a workaround")
        top_n = min(len(feature_importances), len(X_clean.columns))

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


def train_task(labeled_dir, output_dir, task, analyze_errors=True, combined_task=False, combined_file_path=None, target_column=None):
    print(f"[INFO] Starting model training for {'combined' if combined_task else 'individual'} task: {task} target is {target_column}")

    os.makedirs(output_dir, exist_ok=True)

    all_task_metrics = {}

    if combined_task:
        if combined_file_path is None or not os.path.exists(combined_file_path):
            print(f"[ERROR] Combined task file path is required and must exist: {combined_file_path}")
            return {}

        print(f"\n{'=' * 50}")
        print(f"Processing combined task: {task}")
        print(f"{'=' * 50}")

        task_metrics = train_and_evaluate(
            csv_path=combined_file_path,
            task_name=task,
            output_dir=output_dir,
            target_column=target_column)

        if task_metrics:
            all_task_metrics[task] = task_metrics
    elif task == "MarcoPatriciaMerged":
        print(f"Processing task: {task}")

        task_metrics = train_and_evaluate(
            csv_path=labeled_dir,
            task_name=task,
            output_dir=output_dir,
        target_column=target_column)

        if task_metrics:
            all_task_metrics[task] = task_metrics

    else:
        task_folders = [task]
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

            if not os.path.exists(labeled_file_path):
                print(f"[WARN] Labeled features file not found for task {task_name}. Skipping.")
                continue

            task_metrics = train_and_evaluate(
                csv_path=labeled_file_path,
                task_name=task_name,
                output_dir=output_dir,
                target_column=target_column)

            if task_metrics:
                all_task_metrics[task_name] = task_metrics

    # Create summary of all tasks
    #create_all_tasks_summary(all_task_metrics, output_dir)

    return all_task_metrics


def train_baseline_model(labeled_features_file, output_dir=None, target_column='Borg_Cubic',
                         subject_column='subject_id'):
    """
    Train a simple baseline model that predicts the mean RPE value from the training set.
    Uses Leave-One-Subject-Out (LOSO) cross-validation.

    Parameters:
    -----------
    labeled_features_file : str
        Path to the CSV file containing labeled features
    output_dir : str, optional
        Directory to save output files and plots
    target_column : str, default='Borg_Cubic'
        Name of the column containing the target RPE values
    subject_column : str, default='subject_id'
        Name of the column containing subject IDs for LOSO CV

    Returns:
    --------
    dict
        Dictionary containing baseline model performance metrics
    """
    print(f"Loading data from: {labeled_features_file}")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load the labeled features
    try:
        data = pd.read_csv(labeled_features_file)
        print(f"Loaded data with {data.shape[0]} samples and {data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Check if required columns exist
    if target_column not in data.columns:
        print(f"Error: Target column '{target_column}' not found in data")
        return None

    if subject_column not in data.columns:
        print(f"Error: Subject column '{subject_column}' not found in data")
        return None

    # Drop rows with missing target values
    initial_count = data.shape[0]
    data = data.dropna(subset=[target_column])
    if initial_count > data.shape[0]:
        print(f"Dropped {initial_count - data.shape[0]} rows with missing target values")

    # Extract target variable and subject IDs
    y = data[target_column]
    subjects = data[subject_column]

    # Set up arrays to store predictions and actuals
    y_true_all = []
    y_pred_all = []
    subject_ids_all = []
    training_means = []

    # Perform Leave-One-Subject-Out cross-validation
    print("Performing Leave-One-Subject-Out cross-validation...")
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(data, y, subjects):
        # Get training and test data for this fold
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        subjects_test = subjects.iloc[test_idx]

        # Calculate mean RPE from training set (our baseline model)
        mean_rpe = y_train.mean()
        training_means.append(mean_rpe)

        # Predict the mean RPE for all test samples
        y_pred = np.full_like(y_test, mean_rpe)

        # Store actuals and predictions
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        subject_ids_all.extend(subjects_test.tolist())

    # Calculate overall performance metrics
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2 = r2_score(y_true_all, y_pred_all)

    print("Baseline model performance:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

    # Calculate per-subject metrics
    subject_metrics = {}
    unique_subjects = np.unique(subject_ids_all)

    print(f"Calculating metrics for {len(unique_subjects)} individual subjects...")

    for subject in unique_subjects:
        # Get indices for this subject
        subject_indices = [i for i, s in enumerate(subject_ids_all) if s == subject]

        # Get actual and predicted values for this subject
        subject_y_true = [y_true_all[i] for i in subject_indices]
        subject_y_pred = [y_pred_all[i] for i in subject_indices]

        # Calculate metrics
        subject_mae = mean_absolute_error(subject_y_true, subject_y_pred)
        subject_rmse = np.sqrt(mean_squared_error(subject_y_true, subject_y_pred))

        # Store metrics
        subject_metrics[subject] = {
            'mae': subject_mae,
            'rmse': subject_rmse,
            'samples': len(subject_indices)
        }

    # Create a DataFrame for subject-wise metrics
    subject_metrics_df = pd.DataFrame([
        {'subject_id': subject, 'mae': metrics['mae'], 'rmse': metrics['rmse'], 'samples': metrics['samples']}
        for subject, metrics in subject_metrics.items()
    ])

    # Sort by RMSE
    subject_metrics_df = subject_metrics_df.sort_values('rmse')

    # Save results if output directory is specified
    if output_dir:
        # Save predictions and actuals
        results_df = pd.DataFrame({
            'subject_id': subject_ids_all,
            'actual': y_true_all,
            'predicted': y_pred_all,
            'error': np.array(y_true_all) - np.array(y_pred_all)
        })

        results_path = os.path.join(output_dir, 'baseline_predictions.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Saved prediction results to: {results_path}")

        # Save subject metrics
        metrics_path = os.path.join(output_dir, 'baseline_subject_metrics.csv')
        subject_metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved subject metrics to: {metrics_path}")

        # Create scatter plot of actual vs predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true_all, y_pred_all, alpha=0.5)
        plt.plot([min(y_true_all), max(y_true_all)], [min(y_true_all), max(y_true_all)], 'r--')
        plt.xlabel('Actual RPE')
        plt.ylabel('Predicted RPE')
        plt.title('Actual vs Predicted RPE (Baseline Model)')
        plt.savefig(os.path.join(output_dir, 'baseline_scatter_plot.png'), dpi=300)
        plt.close()

        # Create distribution plot of errors
        plt.figure(figsize=(10, 6))
        errors = np.array(y_true_all) - np.array(y_pred_all)
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors (Baseline Model)')
        plt.savefig(os.path.join(output_dir, 'baseline_error_distribution.png'), dpi=300)
        plt.close()

        # Create subject-wise bar chart of RMSE
        plt.figure(figsize=(14, 8))
        sns.barplot(x='subject_id', y='rmse', data=subject_metrics_df)
        plt.xticks(rotation=90)
        plt.xlabel('Subject ID')
        plt.ylabel('RMSE')
        plt.title('RMSE by Subject (Baseline Model)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'baseline_subject_rmse.png'), dpi=300)
        plt.close()

    # Return overall metrics and subject-wise metrics
    return {
        'overall_metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'subject_metrics': subject_metrics,
        'training_means': np.mean(training_means),
        'predictions': {
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'subject_ids': subject_ids_all
        }
    }



import shap
import matplotlib.pyplot as plt
import os


def compute_and_save_shap(model, X, save_dir, task_name, max_display=10, beeswarm_display=10):
    print(f"[INFO] Computing SHAP values for task {task_name}...")
    os.makedirs(save_dir, exist_ok=True)

    # Extract the actual regressor from the pipeline
    if hasattr(model, "named_steps"):
        # Apply the preprocessing steps from the pipeline first
        # (but not the regressor itself)
        preprocessed_X = X
        if 'variance_filter' in model.named_steps:
            preprocessed_X = model.named_steps['variance_filter'].transform(preprocessed_X)
        if 'scaler' in model.named_steps:
            preprocessed_X = model.named_steps['scaler'].transform(preprocessed_X)

        # Convert back to DataFrame with feature names (important for meaningful SHAP plots)
        if hasattr(model.named_steps['variance_filter'], 'get_support'):
            feature_mask = model.named_steps['variance_filter'].get_support()
            selected_features = X.columns[feature_mask]
            preprocessed_X = pd.DataFrame(preprocessed_X, columns=selected_features)

        # Get the regressor
        model_regressor = model.named_steps['regressor']
    else:
        print("[ERROR] Model is not a sklearn pipeline with named steps!")
        return None

    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model_regressor)

    # Use the preprocessed data for SHAP values
    shap_values = explainer.shap_values(preprocessed_X)

    # Save SHAP values as CSV with the correct feature names
    shap_df = pd.DataFrame(shap_values, columns=preprocessed_X.columns)
    shap_csv_path = os.path.join(save_dir, f"shap_values_{task_name}.csv")
    shap_df.to_csv(shap_csv_path, index=False)
    print(f"[INFO] SHAP values saved to {shap_csv_path}")

    # 1. Generate and save SHAP summary bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, preprocessed_X, show=False, plot_type="bar", max_display=max_display)
    shap_summary_plot_path = os.path.join(save_dir, f"shap_summary_bar_{task_name}.png")
    plt.tight_layout()
    plt.savefig(shap_summary_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] SHAP summary bar plot saved to {shap_summary_plot_path}")

    # 2. Generate and save SHAP beeswarm plot with top 10 features
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, preprocessed_X, show=False, plot_type="dot", max_display=beeswarm_display)
    shap_beeswarm_plot_path = os.path.join(save_dir, f"shap_beeswarm_{task_name}.png")
    plt.tight_layout()
    plt.savefig(shap_beeswarm_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] SHAP beeswarm plot saved to {shap_beeswarm_plot_path}")

    # # 3. Optional: Generate and save SHAP waterfall plot for a sample instance
    # # This shows how individual features contribute to a specific prediction
    # if len(preprocessed_X) > 0:
    #     # Select a representative sample (e.g., one with median prediction)
    #     sample_idx = 0  # You could use a more sophisticated selection method
    #     plt.figure(figsize=(10, 8))
    #     shap.plots.waterfall(explainer.expected_value, shap_values[sample_idx],
    #                          features=preprocessed_X.iloc[sample_idx], max_display=10, show=False)
    #     shap_waterfall_path = os.path.join(save_dir, f"shap_waterfall_sample_{task_name}.png")
    #     plt.tight_layout()
    #     plt.savefig(shap_waterfall_path, dpi=300)
    #     plt.close()
    #     print(f"[INFO] SHAP waterfall plot saved to {shap_waterfall_path}")

    return shap_values