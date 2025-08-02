import os
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that selects feature columns based on a YAML config.
    """

    def __init__(self, *, cols_config_path: str):
        """
        Initialize the FeatureSelector with the path to the YAML configuration file.

        Parameters
        ----------
        cols_config_path : str
            Path to the YAML file containing column configurations.
        """
        self.cols_config_path = cols_config_path

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        """
        Fit the FeatureSelector to the data.
        This method loads the column configuration from the YAML file.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to fit the transformer.
        y : None
            Ignored, exists for compatibility with scikit-learn's API.

        Returns
        -------
        self : FeatureSelector
            Fitted transformer.

        Raises
        -------
        FileNotFoundError: If the configuration file does not exist.
        """
        if not os.path.exists(self.cols_config_path):
            raise FileNotFoundError(
                f"Configuration file {self.cols_config_path} does not exist."
            )
        with open(self.cols_config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.feature_columns_ = cfg["feature_columns"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by selecting the feature columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the feature columns specified in the configuration.
        """
        check_is_fitted(self, "feature_columns_")
        return X[self.feature_columns_].copy()


class MissingValueHandler(TransformerMixin, BaseEstimator):
    """
    Transformer that handles missing values:
      - Numeric columns → fillna(0)
      - Object/categorical columns → fillna('unknown')
    """

    def __init__(self):
        """
        Initialize the MissingValueHandler class.
        This class does not require any parameters for initialization.
        """

    def fit(self, X: pd.DataFrame, y=None) -> "MissingValueHandler":
        """
        Fit the MissingValueHandler to the data.
        This method caches the column names for later use in transform.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to fit the transformer.
        y : None
            Ignored, exists for compatibility with scikit-learn's API.

        Returns
        -------
        self : MissingValueHandler
            Fitted transformer.
        """
        self.columns_ = X.columns.to_list()
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.to_list()
        self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.to_list()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by handling missing values.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with missing values handled.
        """
        check_is_fitted(self, ["columns_"])

        # Schema check
        if list(X.columns) != self.columns_:
            raise ValueError("Input columns differ from those seen in fit().")

        # Make a copy to avoid modifying the original DataFrame
        X = X.copy()

        # Fill numeric
        for col in self.numeric_cols_:
            X[col] = X[col].fillna(0)

        # Fill object/categorical
        for col in self.categorical_cols_:
            X[col] = X[col].fillna("unknown")

        return X


class CategoricalFeatureProcessor(TransformerMixin, BaseEstimator):
    """
    Transformer that processes categorical features:
      - Replaces '0.0' and '0' with 'unknown'
      - Replaces special characters with '_'
    """

    def __init__(self):
        """
        Initialize the CategoricalFeatureProcessor class.
        This class does not require any parameters for initialization.
        """

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalFeatureProcessor":
        """
        Fit the CategoricalFeatureProcessor to the data.
        This method caches the column names for later use in transform.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to fit the transformer.
        y : None
            Ignored, exists for compatibility with scikit-learn's API.

        Returns
        -------
        self : CategoricalFeatureProcessor
            Fitted transformer.
        """
        self.columns_ = X.columns.to_list()
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.to_list()
        self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.to_list()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by processing categorical features.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with processed categorical features.
        """
        check_is_fitted(self, "columns_")

        # Make a copy to avoid modifying the original DataFrame
        X = X.copy()

        # Replace '0.0' and '0' with 'unknown' and replace special characters
        for col in self.categorical_cols_:
            X[col] = (
                X[col]
                .replace(["0.0", "0"], "unknown")
                .astype(str)
                .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
            )

        return X
