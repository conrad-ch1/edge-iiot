import os
import click
import pandas as pd
import xgboost as xgb
import mlflow
import optuna
from dotenv import load_dotenv

load_dotenv()

from src.utils.metrics import get_metrics_binary
from src.utils.logger import logger

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "EdgeIIoT_02_Optimization"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def optimize_model_params(
    preprocessed_data_dir: str,
    target_metric: str = "f1",
    target_col: str = "Attack_label",
    n_trials: int = 20,
) -> None:
    """
    Optimize hyperparameters of the XGBoost model using Optuna

    Parameters
    ----------
    preprocessed_data_dir : str
        Directory where the preprocessed data is stored.
    target_metric : str
        Name of the target metric to optimize. Default is "f1".
    target_col : str
        Name of the target column in the dataset. Default is "Attack_label".
    n_trials : int
        Number of trials for hyperparameter optimization. Default is 20.
    """
    # Load the preprocessed data
    logger.info(f"Loading preprocessed data from {preprocessed_data_dir}")
    df_train = pd.read_parquet(f"{preprocessed_data_dir}/train.parquet")
    df_val = pd.read_parquet(f"{preprocessed_data_dir}/val.parquet")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    # Define the hyperparameter search space
    logger.info("Starting hyperparameter optimization")

    def objective(trial):
        # Train the model
        xgb_classifier = xgb.XGBClassifier(
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
            subsample=trial.suggest_float("subsample", 0.2, 1.0),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 1.0),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            random_state=42,
        )
        xgb_classifier.fit(X_train, y_train)

        # Validate the model
        y_pred = xgb_classifier.predict(X_val)
        y_prob = xgb_classifier.predict_proba(X_val)[:, 1]

        # Calculate metrics
        metrics = get_metrics_binary(y_val, y_pred, y_prob)

        with mlflow.start_run(nested=True):
            # Log model parameters
            mlflow.log_params(xgb_classifier.get_params())

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model type
            mlflow.log_param("model_type", "XGB-binary")

        return metrics[target_metric]

    # Create a study and optimize the hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best hyperparameters found: {study.best_params}")


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
@click.option(
    "--target_metric",
    default="f1",
    help="Name of the target metric to optimize.",
)
@click.option(
    "--n_trials",
    default=20,
    help="Number of trials for hyperparameter optimization.",
)
def run(preprocessed_data_dir, target_col, target_metric, n_trials):
    """Command-line interface to run the hyperparameter optimization pipeline."""
    optimize_model_params(
        preprocessed_data_dir=preprocessed_data_dir,
        target_col=target_col,
        target_metric=target_metric,
        n_trials=n_trials,
    )


if __name__ == "__main__":
    run()


# python -m src.pipelines.x03_optuna \
# --preprocessed_data_dir data/processed \
# --target_col Attack_label \
# --target_metric f1 \
# --n_trials 20
