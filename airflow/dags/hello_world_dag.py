import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task

# Paths inside the container
# /opt/airflow
PROJECT_DIR = Path(__file__).resolve().parents[1]
# /opt/airflow/src
SRC_DIR = PROJECT_DIR / "src"
# /opt/airflow/data
DATA_DIR = PROJECT_DIR / "data"
PYTHON_BIN = "python"

# Make src importable
if SRC_DIR.as_posix() not in sys.path:
    sys.path.append(SRC_DIR.as_posix())

COMMON_PREFIX = (
    "set -euo pipefail && "
    f"export PYTHONPATH='{SRC_DIR}:${{PYTHONPATH:-}}' && "
    f"{PYTHON_BIN} "
)

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

with DAG(
    dag_id="xgb_binary_pipeline_bash",
    description="XGBoost binary-classification pipeline via @task.bash",
    start_date=datetime(2023, 10, 1),
    schedule="@daily",  # ← updated arg name
    catchup=False,
    default_args=default_args,
    tags=["edge-iiot", "xgb-binary"],
) as dag:

    # ---------------------- 1. Pre-process data --------------------------- #
    @task.bash
    def preprocess_data():
        return (
            COMMON_PREFIX + f"{PROJECT_DIR}/src/pipelines/x01_preprocess.py "
            f"--input_file {DATA_DIR}/raw/DNN-EdgeIIoT-dataset.csv "
            f"--output_dir {DATA_DIR}/processed "
            f"--cols_config_path {DATA_DIR}/config/valid_columns.yaml "
            f"--target_col Attack_label"
        )

    # ---------------------- 2. Baseline model ----------------------------- #
    @task.bash
    def log_baseline_model():
        return (
            COMMON_PREFIX + f"{PROJECT_DIR}/src/pipelines/x02_log_baseline_model.py "
            f"--preprocessed_data_dir {DATA_DIR}/processed"
        )

    # ---------------------- 3. Optuna tuning ------------------------------ #
    @task.bash
    def optimize_hyperparameters():
        return (
            COMMON_PREFIX + f"{PROJECT_DIR}/src/pipelines/x03_optuna.py "
            f"--preprocessed_data_dir {DATA_DIR}/processed"
        )

    # ---------------------- 4. Register model ----------------------------- #
    @task.bash
    def register_model():
        return (
            COMMON_PREFIX + f"{PROJECT_DIR}/src/pipelines/x04_register_model.py "
            f"--preprocessed_data_dir {DATA_DIR}/processed "
            f"--metric_name f1 --metric_order DESC"
        )

    # Instantiate tasks and define order
    preprocess = preprocess_data()
    baseline = log_baseline_model()
    tune = optimize_hyperparameters()
    register = register_model()

    preprocess >> baseline >> tune >> register
