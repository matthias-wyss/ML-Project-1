import numpy as np


def calculate_accuracy(y_pred, y_true):
    """
    Calculate the accuracy of the predictions.

    Args:
        y_pred: Predicted labels.
        y_true: True labels.

    Returns:
        Accuracy as a float.
    """
    return np.mean(y_pred == y_true)


def calculate_f1_score(y_pred, y_true):
    """
    Calculate the F1 score of the predictions.

    Args:
        y_pred: Predicted labels.
        y_true: True labels.

    Returns:
        F1 score as a float.
    """
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1-Score
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0




