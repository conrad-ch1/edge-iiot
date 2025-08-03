from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the pipeline modules
from src.pipelines.x01_preprocess import preprocess_data
from src.pipelines.x02_log_baseline_model import train_and_log_baseline_model
from src.pipelines.x03_optuna import optimize_model_params
from src.pipelines.x04_register_model import get_best_run, retrain_with_best_params


# Configuration constants
DEFAULT_ARGS = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Data paths (relative to project root)
RAW_DATA_PATH = "data/raw/DNN-EdgeIIoT-dataset.csv"
PROCESSED_DATA_DIR = "data/processed"
CONFIG_PATH = "data/config/valid_columns.yaml"
TARGET_COLUMN = "Attack_label"

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"


def preprocess_task(**context):
    """Task to preprocess the raw data."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    preprocess_data(
        input_file=RAW_DATA_PATH,
        output_dir=PROCESSED_DATA_DIR,
        cols_config_path=CONFIG_PATH,
        target_col=TARGET_COLUMN,
    )


def train_baseline_task(**context):
    """Task to train and log the baseline XGBoost model."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    train_and_log_baseline_model(
        preprocessed_data_dir=PROCESSED_DATA_DIR,
        target_col=TARGET_COLUMN,
    )


def optimize_hyperparameters_task(**context):
    """Task to optimize hyperparameters using Optuna."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    optimize_model_params(
        preprocessed_data_dir=PROCESSED_DATA_DIR,
        target_metric="f1",
        target_col=TARGET_COLUMN,
        n_trials=50,
    )


def register_model_task(**context):
    """Task to register the best model to production."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    best_run = get_best_run(
        metric_name="f1",
        metric_order="DESC",
    )
    retrain_with_best_params(
        preprocessed_data_dir=PROCESSED_DATA_DIR,
        best_run=best_run,
        target_col=TARGET_COLUMN,
    )


def validate_dependencies_task(**context):
    """Task to validate that all required dependencies are available."""
    import mlflow
    import xgboost
    import optuna
    import pandas as pd
    import sklearn

    print("✅ All required dependencies are available")
    print(f"MLflow version: {mlflow.__version__}")
    print(f"XGBoost version: {xgboost.__version__}")
    print(f"Optuna version: {optuna.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")


# Define the DAG
with DAG(
    dag_id="xgb_binary_pipeline",
    default_args=DEFAULT_ARGS,
    description="XGBoost Binary Classification Pipeline for Edge IIoT Dataset",
    start_date=datetime(2025, 8, 1),
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["machine-learning", "xgboost", "binary-classification", "edge-iiot"],
    doc_md="""
    # XGBoost Binary Classification Pipeline
    
    This DAG implements a complete machine learning pipeline for binary classification
    using XGBoost on the Edge IIoT dataset.
    
    ## Pipeline Steps:
    
    1. **Data Preprocessing**: Clean and transform the raw dataset
    2. **Baseline Training**: Train a baseline XGBoost model
    3. **Hyperparameter Optimization**: Use Optuna to find optimal parameters
    4. **Model Registration**: Register the best model for production use
    
    ## Dataset:
    - **Source**: Edge IIoT Dataset
    - **Target**: Attack_label (binary classification)
    - **Features**: Various IoT sensor readings and network metrics
    
    ## MLflow Integration:
    - All experiments are tracked in MLflow
    - Models are versioned and registered
    - Metrics and artifacts are logged for reproducibility
    
    ## Dependencies:
    - MLflow server must be running on localhost:5000
    - Raw dataset must be available in data/raw/
    - Configuration files must be in data/config/
    """,
) as dag:

    # Start task
    start = EmptyOperator(task_id="start", doc_md="Pipeline start marker")

    # Validation task group
    with TaskGroup(group_id="validation") as validation_group:
        validate_deps = PythonOperator(
            task_id="validate_dependencies",
            python_callable=validate_dependencies_task,
            doc_md="Validate that all required Python packages are installed",
        )

    # Data preparation task group
    with TaskGroup(group_id="data_preparation") as data_prep_group:
        preprocess = PythonOperator(
            task_id="preprocess_data",
            python_callable=preprocess_task,
            doc_md="""
            Preprocess the raw Edge IIoT dataset:
            - Feature selection based on configuration
            - Handle missing values
            - Encode categorical variables
            - Split into train/validation/test sets
            - Save processed data as Parquet files
            """,
        )

    # Model training task group
    with TaskGroup(group_id="model_training") as training_group:
        train_baseline = PythonOperator(
            task_id="train_baseline_model",
            python_callable=train_baseline_task,
            doc_md="""
            Train baseline XGBoost model:
            - Use default hyperparameters
            - Log model and metrics to MLflow
            - Create baseline for comparison
            """,
        )

        optimize_params = PythonOperator(
            task_id="optimize_hyperparameters",
            python_callable=optimize_hyperparameters_task,
            doc_md="""
            Optimize XGBoost hyperparameters:
            - Use Optuna for hyperparameter search
            - Optimize F1 score metric
            - Run 20 optimization trials
            - Log all trials to MLflow
            """,
        )

    # Model deployment task group
    with TaskGroup(group_id="model_deployment") as deployment_group:
        register_model = PythonOperator(
            task_id="register_best_model",
            python_callable=register_model_task,
            doc_md="""
            Register the best performing model:
            - Select best model based on F1 score
            - Register model for production use
            - Validate model performance on test set
            """,
        )

    # End task
    end = EmptyOperator(task_id="end", doc_md="Pipeline completion marker")

    # Define task dependencies
    (
        start
        >> validation_group
        >> data_prep_group
        >> training_group
        >> deployment_group
        >> end
    )

    # Within training group, baseline should complete before optimization
    train_baseline >> optimize_params
