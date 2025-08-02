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

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target labels.
    y_pred : np.ndarray
        Estimated targets returned by the classifier.

    Returns
    -------
    dict
        Dictionary containing true positives (tp), false positives (fp),
        false negatives (fn), true negatives (tn), accuracy, precision,
        recall, F1 score, and balanced accuracy score.
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

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target labels.
    y_pred : np.ndarray
        Estimated targets returned by the classifier.
    average : str
        Type of averaging to be performed on the data.
        Default is 'micro' which calculates metrics globally by counting the total
        true positives, false negatives, and false positives.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, F1 score.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
