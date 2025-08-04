FROM apache/airflow:3.0.2

RUN pip install --no-cache-dir \
    apache-airflow==3.0.2 \
    mlflow==3.1.4 \
    xgboost==3.0.2 \
    optuna==4.4.0