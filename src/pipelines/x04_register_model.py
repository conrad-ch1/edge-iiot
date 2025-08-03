import click
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from src.utils.metrics import get_metrics_binary
from src.utils.logger import logger

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME_OPTIMIZATION = "EdgeIIoT_02_Optimization"
EXPERIMENT_NAME_PRODUCTION = "EdgeIIoT_03_Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_best_run(
    metric_name: str,
    metric_order: str,
) -> mlflow.entities.Run:
    """
    Retrieve the best run ID from the MLflow experiment.

    Parameters
    ----------
    metric_name : str
        The name of the metric to look for in the runs.
    metric_order : str
        The order of the metric ("ASC" for ascending, "DESC" for descending).

    Returns
    -------
    str
        The run ID of the best model.
    """
    logger.info("Retrieving the best run from MLflow")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME_OPTIMIZATION)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{metric_name} {metric_order}"],
    )

    if not runs:
        raise ValueError("No runs found in the experiment.")

    best_run = runs[0]

    logger.info(
        f"Best run ID: {best_run.info.run_id}, {metric_name}: {best_run.data.metrics[metric_name]}"
    )

    return best_run


def retrain_with_best_params(
    preprocessed_data_dir: str,
    best_run: mlflow.entities.Run,
    target_col: str = "Attack_label",
) -> None:
    """
    Retrain the model using the best hyperparameters and register it in MLflow.

    Parameters
    ----------
    preprocessed_data_dir : str
        Directory where the preprocessed data is stored.
    best_run : mlflow.entities.Run
        The run object containing the best model parameters.
    target_col : str
        Name of the target column in the dataset. Default is "Attack_label".
    """
    logger.info("Retraining model with best hyperparameters")

    # Load the preprocessed data
    df_train = pd.read_parquet(f"{preprocessed_data_dir}/train.parquet")
    df_val = pd.read_parquet(f"{preprocessed_data_dir}/val.parquet")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    # Get best parameters from the run
    best_params = best_run.data.params

    # Train the model with the best parameters
    xgb_classifier = xgb.XGBClassifier(
        colsample_bytree=float(best_params["colsample_bytree"]),
        subsample=float(best_params["subsample"]),
        learning_rate=float(best_params["learning_rate"]),
        max_depth=int(best_params["max_depth"]),
        random_state=42,
    )
    xgb_classifier.fit(X_train, y_train)

    # Validate the model
    y_pred_xgb = xgb_classifier.predict(X_val)
    y_prob_xgb = xgb_classifier.predict_proba(X_val)[:, 1]

    mlflow.set_experiment(EXPERIMENT_NAME_PRODUCTION)
    with mlflow.start_run(tags={"mlflow.runName": "XGBoostProductionModel"}):
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
            registered_model_name="XGB-binary-production",
        )

        # Log the data pipeline (linking with the logged model in MLflow)
        data_pipeline_file = f"{preprocessed_data_dir}/data_pipeline.joblib"
        mlflow.log_artifact(
            data_pipeline_file,
            artifact_path="data_pipeline",
        )
        logger.info("Model retrained and logged successfully")


@click.command()
@click.option(
    "--preprocessed_data_dir",
    default="data/processed",
    help="Directory where the preprocessed data is stored.",
)
@click.option(
    "--metric_name",
    default="f1",
    help="Name of the metric to look for in the runs.",
)
@click.option(
    "--metric_order",
    default="DESC",
    type=click.Choice(["ASC", "DESC"], case_sensitive=True),
    help="Order of the metric (ASC for ascending, DESC for descending).",
)
def run(preprocessed_data_dir, metric_name, metric_order):
    """Command-line interface to run the model registration pipeline."""
    best_run = get_best_run(metric_name=metric_name, metric_order=metric_order)
    retrain_with_best_params(
        preprocessed_data_dir=preprocessed_data_dir,
        best_run=best_run,
        target_col="Attack_label",
    )


if __name__ == "__main__":
    run()

# python -m src.pipelines.x04_register_model \
# --preprocessed_data_dir data/processed \
# --metric_name f1 \
# --metric_order DESC
