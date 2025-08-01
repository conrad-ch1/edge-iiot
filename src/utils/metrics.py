import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


def get_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return basic evaluation metrics for a binary classification task.

    :param: y_true: Ground truth (correct) target labels.
    :param: y_pred: Estimated targets returned by the classifier.
    :returns: Dictionary containing accuracy, precision, recall, F1 score,
        and the confusion matrix (as a nested list for easy JSON export).
    """
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    tn = sum((y_true == 0) & (y_pred == 0))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def get_metrics_multiclass(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "micro"
) -> dict:
    """Return basic evaluation metrics for a multiclass classification task.

    :param: y_true: Ground truth (correct) target labels.
    :param: y_pred: Estimated targets returned by the classifier.
    :param average: Type of averaging to be performed on the data.
        Default is 'micro' which calculates metrics globally by counting the total
        true positives, false negatives, and false positives.
    :returns: Dictionary containing accuracy, precision, recall, F1 score,
        and the confusion matrix (as a nested list for easy JSON export).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
