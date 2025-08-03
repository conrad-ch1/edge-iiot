import click
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.utils.data_processing import (
    FeatureSelector,
    MissingValueHandler,
    CategoricalFeatureProcessor,
    CustomDictVectorizer,
)
from src.utils.logger import logger


def preprocess_data(
    input_file: str,
    output_dir: str,
    cols_config_path: str,
    target_col: str = "Attack_label",
) -> None:
    """
    Preprocess the dataset by selecting features, handling missing values,
    and encoding categorical variables. Saves the processed data to parquet files.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file containing the dataset.
    output_dir : str
        Directory where the preprocessed data will be saved.
    cols_config_path : str
        Path to the YAML file containing column configurations.
    target_col : str
        Name of the target column in the dataset. Default is "Attack_label".
    """
    # Load the raw data
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file, low_memory=False)

    logger.info("Data preprocessing...")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Build the data pipeline
    data_pipeline = Pipeline(
        steps=[
            ("feature_selector", FeatureSelector(cols_config_path=cols_config_path)),
            ("missing_value_handler", MissingValueHandler()),
            ("categorical_feature_processor", CategoricalFeatureProcessor()),
            ("custom_dict_vectorizer", CustomDictVectorizer()),
        ]
    )

    # Split the dataset into training (0.6), validation (0.2), and test sets (0.2)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(
        f"Train shape: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # Fit the pipeline on the training data
    data_pipeline.fit(X_train)
    X_train_transformed = data_pipeline.transform(X_train)
    X_val_transformed = data_pipeline.transform(X_val)
    X_test_transformed = data_pipeline.transform(X_test)

    # Save the transformed datasets to parquet files
    logger.info(f"Saving preprocessed data to {output_dir}")
    df_train = pd.concat([X_train_transformed, y_train], axis=1)
    df_val = pd.concat([X_val_transformed, y_val], axis=1)
    df_test = pd.concat([X_test_transformed, y_test], axis=1)

    df_train.to_parquet(f"{output_dir}/train.parquet")
    df_val.to_parquet(f"{output_dir}/val.parquet")
    df_test.to_parquet(f"{output_dir}/test.parquet")

    # Save the preprocessing pipeline
    joblib.dump(data_pipeline, f"{output_dir}/data_pipeline.joblib")

    logger.info("Data saved successfully")


@click.command()
@click.option(
    "--input_file",
    default="data/raw/DNN-EdgeIIoT-dataset.csv",
    help="Path to the input CSV file containing the dataset.",
)
@click.option(
    "--output_dir",
    default="data/processed",
    help="Directory where the preprocessed data will be saved.",
)
@click.option(
    "--cols_config_path",
    default="data/config/valid_columns.yaml",
    help="Path to the YAML file containing column configurations.",
)
@click.option(
    "--target_col",
    default="Attack_label",
    help="Name of the target column in the dataset.",
)
def run(input_file, output_dir, cols_config_path, target_col):
    """Command-line interface to run the data preprocessing pipeline."""
    preprocess_data(input_file, output_dir, cols_config_path, target_col)


if __name__ == "__main__":
    run()

# python -m src.pipelines.x01_preprocess \
# --input_file data/raw/DNN-EdgeIIoT-dataset.csv \
# --output_dir data/processed --cols_config_path data/config/valid_columns.yaml \
# --target_col Attack_label
