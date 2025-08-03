import pandas as pd
import xgboost as xgb
import mlflow
import optuna

from src.utils.metrics import get_metrics_binary
from src.utils.logger import logger

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("2_hyperparameter_optimization")


def optimize_model_params(
    preprocessed_data_dir: str,
    target_metric: str = "f1",
    target_col: str = "Attack_label",
) -> None:
    # Load the preprocessed data
    logger.info("Loading preprocessed data from %s", preprocessed_data_dir)
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
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.8),
            subsample=trial.suggest_float("subsample", 0.2, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            gamma=trial.suggest_float("gamma", 0.0, 10.0),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 15.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 15.0),
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            random_state=42,
            tree_method="hist",
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
    study.optimize(objective, n_trials=20, n_jobs=-1)
    logger.info("Best hyperparameters found: %s", study.best_params)


if __name__ == "__main__":
    optimize_model_params(
        "data/processed", target_col="Attack_label", target_metric="f1"
    )
