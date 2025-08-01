FROM python:3.12-slim

# Install dependencies
RUN pip install --no-cache-dir mlflow>=3.1.4

# Expose the MLflow server port
EXPOSE 5000

# Create a directory for MLflow artifacts
RUN mkdir -p /mlflow

# Set the working directory
WORKDIR /mlflow

# Set command to run the MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///backend.db", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]