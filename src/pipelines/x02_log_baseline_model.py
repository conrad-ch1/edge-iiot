import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature

from src.utils.metrics import get_metrics_binary
from src.utils.logger import logger


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("1_baseline_model")


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
    target : str
        Name of the target column in the dataset. Default is "Attack_label".
    """
    # Load the preprocessed data
    logger.info("Loading preprocessed data from %s", preprocessed_data_dir)
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
        )

        # Log the data pipeline (linking with the logged model in MLflow)
        data_pipeline_file = f"{preprocessed_data_dir}/data_pipeline.joblib"
        mlflow.log_artifact(
            data_pipeline_file,
            artifact_path="data_pipeline",
        )
        logger.info("Model and metrics logged successfully")


if __name__ == "__main__":
    train_and_log_baseline_model(
        preprocessed_data_dir="data/processed",
        target="Attack_label",
    )
