import click
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature

from src.utils.metrics import get_metrics_binary
from src.utils.logger import logger

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "EdgeIIoT_02_Baseline"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def train_and_log_baseline_model(
    preprocessed_data_dir: str,
    target_col: str = "Attack_label",
) -> None:
    """
    Train and log a baseline model using the preprocessed data.

    Parameters
    ----------
    preprocessed_data_dir : str
        Directory where the preprocessed data is stored.
    target_col : str
        Name of the target column in the dataset. Default is "Attack_label".
    """
    # Load the preprocessed data
    logger.info(f"Loading preprocessed data from {preprocessed_data_dir}")
    df_train = pd.read_parquet(f"{preprocessed_data_dir}/train.parquet")
    df_val = pd.read_parquet(f"{preprocessed_data_dir}/val.parquet")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    # XGBoost Classifier
    logger.info("Training XGBoost Classifier")
    xgb_classifier = xgb.XGBClassifier(random_state=42)
    xgb_classifier.fit(X_train, y_train)
    y_pred_xgb = xgb_classifier.predict(X_val)
    y_prob_xgb = xgb_classifier.predict_proba(X_val)[:, 1]

    with mlflow.start_run(tags={"mlflow.runName": "XGBoostBinaryClassification"}):
        # Log model parameters
        mlflow.log_params(xgb_classifier.get_params())

        # Log metrics
        metrics = get_metrics_binary(y_val, y_pred_xgb, y_prob_xgb)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        mlflow.log_metrics(metrics)

        # Log model type
        mlflow.log_param("model_type", "XGB-binary")

        # Infer signature from data
        signature = infer_signature(X_val, y_pred_xgb)

        # Log the model
        mlflow.xgboost.log_model(
            xgb_classifier,
            name="xgb_classifier",
            model_format="ubj",
            signature=signature,
            registered_model_name="XGB-binary-baseline",
        )

        # Log the data pipeline (linking with the logged model in MLflow)
        data_pipeline_file = f"{preprocessed_data_dir}/data_pipeline.joblib"
        mlflow.log_artifact(
            data_pipeline_file,
            artifact_path="data_pipeline",
        )
        logger.info("Model and metrics logged successfully")


@click.command()
@click.option(
    "--preprocessed_data_dir",
    default="data/processed",
    help="Directory where the preprocessed data is stored.",
)
@click.option(
    "--target_col",
    default="Attack_label",
    help="Name of the target column in the dataset.",
)
def run(preprocessed_data_dir, target_col):
    """Command-line interface to run the baseline model training and logging."""
    train_and_log_baseline_model(
        preprocessed_data_dir=preprocessed_data_dir,
        target_col=target_col,
    )


if __name__ == "__main__":
    run()

# python -m src.pipelines.x02_log_baseline_model \
# --preprocessed_data_dir data/processed \
# --target_col Attack_label
