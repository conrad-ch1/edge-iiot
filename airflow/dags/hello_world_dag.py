from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import mlflow


# Function to be executed by the PythonOperator
def hello_world():
    print("Hello, World!")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='hello_world_dag',
    default_args=default_args,
    description='A simple hello world DAG',
    start_date=datetime(2023, 10, 1),
    schedule='@daily',
    catchup=False,
    tags=['example', 'hello_world'],
) as dag:

    # Define the start task
    start_task = EmptyOperator(task_id='start')
    
    # Define the PythonOperator task
    hello_task = PythonOperator(
        task_id='hello_world_task',
        python_callable=hello_world,
    )
    # Define the end task    
    end_task = EmptyOperator(task_id='end')

    # Set task dependencies
    (start_task >> hello_task >> end_task)