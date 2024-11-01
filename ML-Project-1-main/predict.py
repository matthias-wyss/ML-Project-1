import numpy as np
import random
from implementations import *
from preprocess_data import *
from cross_validation import *
from utils import *
from predict import *

def predict(w, x, method, threshold=0.5):
    """
    Generate predictions based on the model weights and input features.

    Args:
        w: Weights of the model.
        x: Input features.
        method: The method used for prediction.
        threshold: Threshold for classifying as 1 (default is 0)

    Returns:
        Predicted labels based on the specified method.
    """
    if method in ["logistic_regression", "reg_logistic_regression"]:
        # Compute predicted probabilities for logistic regression
        y_pred_prob = sigmoid(x @ w)
        return (y_pred_prob >= threshold).astype(int)  # Binary classification
    else:
        # Compute raw predictions for other methods
        raw_predictions = x @ w
        return (raw_predictions >= threshold).astype(int)



def count_unique_labels(y_pred):
    """
    Count and print occurrences of each unique label in predictions.

    Args:
        y_pred: numpy.ndarray, predicted labels.

    Returns:
        None
    """
    unique_labels, counts = np.unique(y_pred, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        print(f"Number of predicted labels {label}: {count}")