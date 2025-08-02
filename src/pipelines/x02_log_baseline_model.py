import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow

from src.utils.metrics import get_metrics_binary

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("1_baseline_model")


def train_and_log_baseline_model(
    preprocessed_data_dir: str,
) -> None:
    """
    Train and log a baseline model using the preprocessed data.

    Parameters
    ----------
    train_file : str
        Path to the training data (features).
    train_target_file : str
        Path to the training data (target).
    val_file : str
        Path to the validation data (features).
    val_target_file : str
        Path to the validation data (target).
    """
    # Load the preprocessed data
    X_train = pd.read_parquet(f"{preprocessed_data_dir}/X_train.parquet")
    y_train = pd.read_parquet(f"{preprocessed_data_dir}/y_train.parquet")
    X_val = pd.read_parquet(f"{preprocessed_data_dir}/X_val.parquet")
    y_val = pd.read_parquet(f"{preprocessed_data_dir}/y_val.parquet")

    # XGBoost Classifier
    xgb_classifier = xgb.XGBClassifier(random_state=42)
    xgb_classifier.fit(X_train, y_train)
    y_pred_xgb = xgb_classifier.predict(X_val)
    y_prob_xgb = xgb_classifier.predict_proba(X_val)[:, 1]

    with mlflow.start_run(tags={"mlflow.runName": "XGBoostBinaryClassification"}):
        # Log model parameters
        mlflow.log_params(xgb_classifier.get_params())

        # Log metrics
        metrics = get_metrics_binary(y_val, y_pred_xgb)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        mlflow.log_metrics(metrics)

        # Calculate and log ROC AUC
        roc_auc = roc_auc_score(y_val, y_prob_xgb)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model type
        mlflow.log_param("model_type", "XGB-binary")

        # Log the model
        mlflow.xgboost.log_model(
            xgb_classifier,
            name="xgb_classifier",
            input_example=X_val.iloc[:5],
        )

        # Log the data pipeline (linking with the logged model in MLflow)
        data_pipeline_file = f"{preprocessed_data_dir}/data_pipeline.joblib"
        mlflow.log_artifact(
            data_pipeline_file,
            artifact_path="data_pipeline",
        )
