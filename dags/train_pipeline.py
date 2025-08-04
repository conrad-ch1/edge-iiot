import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task

################ Configuration ################
# Paths inside the container
# /opt/airflow
PROJECT_DIR = Path(__file__).resolve().parents[1]
# /opt/airflow/data
DATA_DIR = PROJECT_DIR / "data"
PYTHON_BIN = "python"

COMMON_PREFIX = f"cd {PROJECT_DIR} && {PYTHON_BIN} -m "
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}
##############################################

with DAG(
    dag_id="xgb_binary_pipeline_bash",
    description="XGBoost binary-classification pipeline via @task.bash",
    start_date=datetime(2025, 8, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["edge-iiot", "xgb-binary"],
) as dag:

    @task.bash
    def preprocess_data():
        return (
            COMMON_PREFIX + "src.pipelines.x01_preprocess "
            f"--input_file {DATA_DIR}/raw/DNN-EdgeIIoT-dataset.csv "
            f"--output_dir {DATA_DIR}/processed "
            f"--holdout_dir {DATA_DIR}/raw "
            f"--cols_config_path {DATA_DIR}/config/valid_columns.yaml "
            f"--target_col Attack_label"
        )

    @task.bash
    def log_baseline_model():
        return (
            COMMON_PREFIX + "src.pipelines.x02_log_baseline_model "
            f"--preprocessed_data_dir {DATA_DIR}/processed"
        )

    @task.bash
    def optimize_hyperparameters():
        return (
            COMMON_PREFIX + "src.pipelines.x03_optuna "
            f"--preprocessed_data_dir {DATA_DIR}/processed"
        )

    @task.bash
    def register_model():
        return (
            COMMON_PREFIX + "src.pipelines.x04_register_model "
            f"--preprocessed_data_dir {DATA_DIR}/processed "
            "--metric_name f1 --metric_order DESC"
        )

    # Instantiate tasks and define order
    preprocess = preprocess_data()
    baseline = log_baseline_model()
    tune = optimize_hyperparameters()
    register = register_model()

    preprocess >> baseline >> tune >> register