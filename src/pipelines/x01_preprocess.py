import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

from src.utils.data_processing import (
    FeatureSelector,
    MissingValueHandler,
    CategoricalFeatureProcessor,
)


def preprocess_data(
    input_file: str,
    output_dir: str,
    cols_config_path: str,
    target: str = "Attack_label",
) -> None:
    """
    Preprocess the dataset by selecting features, handling missing values,
    and encoding categorical variables.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file containing the dataset.
    output_dir : str
        Directory where the preprocessed data will be saved.
    cols_config_path : str
        Path to the YAML file containing column configurations.
    target : str
        Name of the target column in the dataset. Default is "Attack_label".
    """
    # Load the raw data
    df = pd.read_csv(input_file)

    y = df[target]
    X = df.drop(columns=[target])

    # Build the data pipeline
    data_pipeline = Pipeline(
        steps=[
            ("feature_selector", FeatureSelector(cols_config_path=cols_config_path)),
            ("missing_value_handler", MissingValueHandler()),
            ("categorical_feature_processor", CategoricalFeatureProcessor()),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Split the dataset into training (0.6), validation (0.2), and test sets (0.2)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Fit the pipeline on the training data
    data_pipeline.fit(X_train)
    X_train_transformed = data_pipeline.transform(X_train)
    X_val_transformed = data_pipeline.transform(X_val)
    X_test_transformed = data_pipeline.transform(X_test)

    # Save the transformed datasets to parquet files
    X_train_transformed.to_parquet(f"{output_dir}/X_train.parquet")
    y_train.to_parquet(f"{output_dir}/y_train.parquet")
    X_val_transformed.to_parquet(f"{output_dir}/X_val.parquet")
    y_val.to_parquet(f"{output_dir}/y_val.parquet")
    X_test_transformed.to_parquet(f"{output_dir}/X_test.parquet")
    y_test.to_parquet(f"{output_dir}/y_test.parquet")

    # Save the preprocessing pipeline
    joblib.dump(data_pipeline, f"{output_dir}/data_pipeline.joblib")

    print("Data preprocessing completed and saved to parquet files.")
