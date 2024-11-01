import numpy as np
from preprocess_data import *
from implementations import *
from predict import *
from helpers import *


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

def predict_with_method(
    x_train,
    y_train,
    x_test,
    test_ids,
    method,
    lambda_=0.1,
    initial_w=None,
    max_iters=1000,
    gamma=0.0001,
    replace_nan_by=-1,
    column_nan_threshold=0.01,
    row_nan_threshold=1,
    continuous_threshold=0,
    normalization_method="standardize",
    outliers=None,
    z_score_threshold=3,
    max_false_percentage=0.3,
    balance_method="random_upsampling",
    target_minority_ratio=0.2,
    noise_ratio=0.02,
    add_bias=False,
    pca_ratio=0.95,
    decision_threshold=0.2,
    preprocess_verbose=False
):
    """
    Predict labels for the test dataset using a specified regression method.

    Args:
        x_train: numpy.ndarray, training input features.
        y_train: numpy.ndarray, training target labels.
        x_test: numpy.ndarray, testing input features.
        test_ids: numpy.ndarray, IDs for test samples.
        method: str, the regression method to use.
        lambda_: float, regularization parameter (default is 0.1).
        initial_w: numpy.ndarray, initial weights for gradient descent.
        max_iters: int, maximum iterations for gradient descent (default is 1000).
        gamma: float, learning rate for gradient descent (default is 0.01).
        replace_nan_by: int, number to replace NaNs by.
        column_nan_threshold: threshold for removing features with too many NaN values.
        row_nan_threshold: threshold for removing samples with too many NaN values.
        continuous_threshold: threshold for determining continuous features.
        normalization_method: method to normalize features.
        outliers: strategy for handling outliers.
        z_score_threshold: threshold for identifying outliers based on Z-scores.
        max_false_percentage: maximum allowable percentage of false samples.
        balance_method: method for balancing the dataset.
        target_minority_ratio: ratio for balancing the target classes.
        add_bias: boolean, whether to add a bias term to the model.
        pca_ratio: float, ratio of number of features after PCA / number of features before PCA.
        decision_threshold: float, threshold for classifying as 1.
        preprocess_verbose: boolean, if True, print preprocessing steps.

    Returns:
        numpy.ndarray, a 2D array of IDs and predicted labels.
    """
    w = np.zeros(x_train.shape[1])

    # Preprocess training and test data
    if preprocess_verbose:
        print("Preprocessing data...")
    
    preprocessed_x_train, preprocessed_x_test, preprocessed_y_train = preprocess_global(
        x_train, x_test, y_train
    )

    # Train the model using the specified method
    if method == "mean_squared_error_gd":
        w, _ = mean_squared_error_gd(preprocessed_y_train, preprocessed_x_train, initial_w, max_iters, gamma)
    elif method == "mean_squared_error_sgd":
        w, _ = mean_squared_error_sgd(preprocessed_y_train, preprocessed_x_train, initial_w, max_iters, gamma)
    elif method == "least_squares":
        w, _ = least_squares(preprocessed_y_train, preprocessed_x_train)
    elif method == "ridge_regression":
        w, _ = ridge_regression(preprocessed_y_train, preprocessed_x_train, lambda_)
    elif method == "logistic_regression":
        w, _ = logistic_regression(preprocessed_y_train, preprocessed_x_train, initial_w, max_iters, gamma)
    elif method == "reg_logistic_regression":
        w, _ = reg_logistic_regression(preprocessed_y_train, preprocessed_x_train, lambda_, initial_w, max_iters, gamma)
    else:
        raise ValueError("Invalid method specified.")

    # Make predictions on the test set using the helper function
    y_pred = predict(w, preprocessed_x_test, method, decision_threshold)

    # Map 0 to -1 for binary classification
    y_pred = np.where(y_pred == 0, -1, y_pred)

    # Count occurrences of each unique label and print them
    count_unique_labels(y_pred)

    # Use the create_csv_submission function to save predictions
    create_csv_submission(test_ids, y_pred, "predictions.csv")

    return np.column_stack((test_ids, y_pred))




