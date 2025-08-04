# app.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import datetime
import pandas as pd
import duckdb
import joblib, xgboost as xgb, mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Model and pipeline loading
MODEL_VERSION = "1"
MODEL_NAME = "xgb_classifier"

client = mlflow.MlflowClient()
model_info = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)

artifacts_base = f"models/{MODEL_VERSION}/artifacts"

mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{model_info.run_id}/artifacts/model.ubj",
    dst_path=artifacts_base,
)

mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{model_info.run_id}/data_pipeline/data_pipeline.joblib",
    dst_path=artifacts_base,
)

pipe = joblib.load(f"{artifacts_base}/data_pipeline.joblib")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"{artifacts_base}/model.ubj")

# Persistence layer
DUCK_PATH = "inference_log.duckdb"
duckdb.sql(f"CREATE TABLE IF NOT EXISTS predictions AS SELECT 0 AS dummy").execute()
duckdb.sql("DELETE FROM predictions").execute()

# Api configuration
app = FastAPI(title="XGB-Binary Inference")


class Record(BaseModel):
    """Record for inference request."""

    features: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.features])


@app.post("/predict")
def predict(rec: Record, background_tasks: BackgroundTasks):
    df = rec.to_dataframe()
    ground_truth = df.pop("Attack_label", pd.NA)

    # Prediction
    X = pipe.transform(df)
    y_pred = xgb_model.predict(X)
    predicted_label = y_pred[0]

    # Log prediction
    background_tasks.add_task(log_prediction, df, predicted_label, ground_truth)
    return {"prediction": predicted_label}


def log_prediction(features_df: pd.DataFrame, pred: int, gt: int):
    log_row = features_df.copy()
    log_row["ground_truth"] = gt
    log_row["predicted_label"] = pred
    log_row["ts"] = datetime.datetime.now()

    duckdb.append("predictions", log_row)
