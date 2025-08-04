# Edge-IIoT Dataset Project

## Overview

**Edge-IIoTset** is a realistic cyber-security dataset for Industrial & Edge IoT intrusion detection systems. This project provides a complete ML pipeline for training and deploying intrusion detection models using the Edge-IIoTset dataset.

### Dataset Description

Edge-IIoTset is a public benchmark designed for training and evaluating intrusion-detection models that must operate close to the physical process—either on resource-constrained edge nodes or in federated learning settings.

Unlike cloud-centric corpora that capture traffic only after cleaning and uplink, Edge-IIoTset is collected in-situ on a seven-layer industrial testbed spanning:

- **Cloud Computing ↔ NFV ↔ Blockchain ↔ Fog ↔ SDN ↔ Edge ↔ IoT/IIoT** Perception layers
- Implementation with production-grade open-source stacks (ThingsBoard, OPNFV, Hyperledger Sawtooth, Eclipse Ditto, ONOS, Mosquitto)
- **10+ sensor/actuator types**: ultrasonic, flame, pH, Modbus slaves, soil-moisture, heart-rate, etc.
- **Multi-protocol traffic**: MQTT, Modbus/TCP, HTTP, ICMP, ARP captured as raw PCAP and CSV/Parquet flows
- **61 high-correlation features** distilled from 1,176 candidates using Zeek & TShark

### Dataset Contents

| Aspect | Details |
|--------|---------|
| **Normal traces** | 10 PCAP/CSV pairs (≈ 9M records) covering typical sensor telemetry and Modbus operations |
| **Attack surface** | 14 curated attacks grouped into 5 threat families |
| **Attack families** | • DoS/DDoS (SYN, UDP, HTTP, ICMP floods)<br>• Information Gathering (port scan, OS fingerprint, vulnerability scan)<br>• Man-in-the-Middle (DNS & ARP spoofing)<br>• Injection (XSS, SQL-i, malicious upload)<br>• Malware (backdoor, password-crack, ransomware) |
| **Labels** | Attack_label (binary) and Attack_type (multiclass) ready for ML pipelines |
| **Learning modes** | Reference baselines for Random Forest, SVM, XGBoost and DNN in both centralized and federated scenarios |
| **License & DOI** | CC-BY-4.0 – DOI 10.36227/techrxiv.18857336.v1 |

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Git
- Make

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd edge-iiot

# See all available commands
make help
```

### 2. Download Dataset

```bash
# Download the Edge-IIoT dataset
make data

# Verify dataset is available
make check-data
```

## 🔧 Development Environment Setup


### Using UV (Modern Python Package Manager)

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🐳 Running with Docker

### Start Infrastructure Services

```bash
# Set Airflow UID for Docker
echo -e "AIRFLOW_UID=$(id -u)" > .env
```
add MLFLOW_TRACKING_URI=http://localhost:5000 to .env file

# Start all services (PostgreSQL, MLflow, Airflow)
docker compose up -d

# View logs
docker compose logs -f
```

### Access Web Interfaces

- **Airflow UI**: http://localhost:8080 (admin/admin)
- **MLflow UI**: http://localhost:5000
- **FastAPI Application**: http://localhost:8000 (when running)

## 🎯 Running the ML Pipeline

### Option 1: Using Airflow (Orchestrated Pipeline)

1. **Start Airflow**:
   ```bash
   docker compose up -d
   
2. **Access Airflow UI**: http://localhost:8080
   - Username: `airflow`
   - Password: `airflow`

3. **Trigger the Pipeline**:
   - Find the DAG: `xgb_binary_pipeline_bash`
   - Click "Trigger DAG" to start the training pipeline

4. **Monitor Progress**: Watch the pipeline execution in the Airflow UI

### Option 2: Manual Pipeline Execution
```bash
# Activate environment
source venv/bin/activate  # or source .venv/bin/activate for uv

# 1. Preprocess data
python -m src.pipelines.x01_preprocess \
  --input_file data/raw/DNN-EdgeIIoT-dataset.csv \
  --output_dir data/processed \
  --holdout_dir data/raw \
  --cols_config_path data/config/valid_columns.yaml \
  --target_col Attack_label

# 2. Log baseline model
python -m src.pipelines.x02_log_baseline_model \
  --train_file data/processed/train.parquet \
  --val_file data/processed/val.parquet

# 3. Hyperparameter optimization
python -m src.pipelines.x03_optuna \
  --train_file data/processed/train.parquet \
  --val_file data/processed/val.parquet

# 4. Register best model
python -m src.pipelines.x04_register_model
```

### Option 3: Start MLflow Server Only

```bash
# Start MLflow tracking server
make mlflow-server

# Access MLflow UI at http://localhost:5000
```

## 🚀 Deploying the Model

### Start the FastAPI Application

```bash
# Ensure MLflow server is running
make mlflow-server

# In another terminal, start the FastAPI app
cd deployment
python app.py

# API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

## 🔍 Development Tools

### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Run tests
make test

# Run all CI checks
make ci
```

### Project Information

```bash
# Show project info and status
make info

# Clean temporary files
make clean

# Clean dataset files
make clean-data
```

## 📁 Project Structure

```
├── dags/                    # Airflow DAGs
│   └── train_pipeline.py    # Main training pipeline
├── src/                     # Source code
│   ├── pipelines/           # ML pipeline steps
│   └── utils/               # Utility functions
├── deployment/              # FastAPI application
│   └── app.py              # Model serving API
├── data/                    # Dataset storage
│   ├── raw/                # Raw dataset files
│   ├── processed/          # Processed data
│   └── config/             # Configuration files
├── notebooks/               # Jupyter notebooks
├── docker/                  # Docker configurations
├── logs/                    # Airflow logs
├── mlflow-artifacts/        # MLflow artifacts
├── docker-compose.yaml      # Docker services
├── Makefile                # Build automation
└── pyproject.toml          # Project configuration
```

## 📊 Pipeline Steps

1. **Data Preprocessing** (`x01_preprocess.py`):
   - Loads raw CSV dataset
   - Performs data cleaning and feature engineering
   - Splits data into train/validation sets
   - Saves processed data as Parquet files

2. **Baseline Model** (`x02_log_baseline_model.py`):
   - Trains baseline XGBoost model
   - Logs metrics and model to MLflow

3. **Hyperparameter Optimization** (`x03_optuna.py`):
   - Uses Optuna for hyperparameter tuning
   - Optimizes XGBoost parameters
   - Logs best model to MLflow

4. **Model Registration** (`x04_register_model.py`):
   - Registers the best model in MLflow Model Registry
   - Prepares model for deployment

## 🛠️ Available Make Commands

Run `make help` to see all available commands:

```bash
make help           # Show help message
make data           # Download the Edge-IIoT dataset
make check-data     # Check if dataset is available
make setup          # Set up development environment
make install        # Install project dependencies
make dev-install    # Install development dependencies
make mlflow-server  # Start MLflow server
make lint           # Run code linting
make format         # Format code
make test           # Run tests
make docs           # Generate documentation
make info           # Show project information
make clean          # Clean temporary files
make clean-data     # Remove downloaded dataset
make ci             # Run CI checks
make all            # Run all setup and verification tasks
```

## 🔧 Troubleshooting

### Common Issues

1. **Docker permission errors**:
   ```bash
   # Ensure .env file has correct UID
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```

2. **MLflow connection issues**:
   ```bash
   # Make sure MLflow server is running
   make mlflow-server
   ```

3. **Dataset download issues**:
   ```bash
   # Clean and retry download
   make clean-data
   make data
   ```

4. **Virtual environment issues**:
   ```bash
   # Remove and recreate environment
   rm -rf venv
   make setup
   make install
   ```

## 📚 Additional Resources

- **Dataset Paper**: [Edge-IIoTset DOI](https://doi.org/10.36227/techrxiv.18857336.v1)
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Airflow Documentation**: https://airflow.apache.org/docs/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make ci`
5. Submit a pull request

## 📄 License

This project is licensed under CC-BY-4.0. See the dataset documentation for more details.
