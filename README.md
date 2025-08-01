# edge-iiot

# See all available commands
make help

# Download the dataset
make data

# Set up development environment
make setup

# Check if data is ready
make check-data

# Run all setup tasks
make all







echo -e "AIRFLOW_UID=$(id -u)" > .env