FROM apache/airflow:3.0.2

RUN pip install --no-cache-dir apache-airflow==3.0.2
RUN pip install --no-cache-dir mlflow==3.1.4