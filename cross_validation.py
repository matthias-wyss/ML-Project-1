import numpy as np
import random
from implementations import *
from preprocess_data import *
from utils import *
from predict import *
import pickle

def k_fold_split(x, y, k, seed=1):
    """
    Split data into K folds for cross-validation.
    
    Args:
        x: numpy array of shape (N, D), where N is the number of samples and D is the number of features.
        y: numpy array of shape (N,), representing the labels corresponding to the samples.
        k: integer, the number of folds for cross-validation.
        seed: integer, random seed for reproducibility of the split.
    
    Returns:
        folds_x: list of numpy arrays containing the K folds for features.
        folds_y: list of numpy arrays containing the K folds for labels.
    """
    np.random.seed(seed)  # Set random seed for reproducibility
    N = len(x)  # Get the number of samples
    indices = np.random.permutation(N)  # Randomly permute indices for shuffling the data
    
    # Split indices into K approximately equal parts
    folds_x = np.array_split(x[indices], k)  # Split features into K folds
    folds_y = np.array_split(y[indices], k)  # Split labels into K folds
    
    return folds_x, folds_y  # Return the feature and label folds

def final_k_folds(x_train, y_train, k, method):
    """
    The goal of this function is to preprocess and prepare all the combinations of folds one and for all, so as to not to re-do them multiple times inside the cross-validation function.
    """
    folds_x, folds_y = k_fold_split(x_train, y_train, k)
    preprocessed_folds = []
    for i in range(k):
        # Use the i-th fold as the test set and the remaini  ng as the training set
        x_te = folds_x[i]
        y_te = folds_y[i]
        
        # Convert labels from -1 to 0 for compatibility with the model
        y_te = convert_labels(y_te)

        # Combine the remaining folds to create the training set
        x_tr = np.concatenate([folds_x[j] for j in range(k) if j != i])
        y_tr = np.concatenate([folds_y[j] for j in range(k) if j != i])
        
        print(f'Preprocessing for fold {i+1}:\n')

        # Preprocess training and test data separately (fit on train, apply to test)
        preprocessed_x_train, preprocessed_x_test, preprocessed_y_train = preprocess(x_tr, x_te, y_tr, method)

        #print("Mean of each feature in x_train:", np.mean(preprocessed_x_train, axis=0))
        #print("Standard deviation of each feature in x_test:", np.std(preprocessed_x_train, axis=0))
        #print("Mean of each feature in x_test:", np.mean(preprocessed_x_test, axis=0))
        #print("Standard deviation of each feature in x_train:", np.std(preprocessed_x_test, axis=0))


        # Fill the Kx4 matrix of the K-tuples of data the cross validation will train on
        preprocessed_folds.append([preprocessed_x_train, preprocessed_y_train, preprocessed_x_test, y_te])
        
    return preprocessed_folds

def cross_validation_step(preprocessed_x_train, preprocessed_y_train, x_test, y_test, method, lambda_, initial_w, gamma, decision_threshold= 0.2, max_iters = 1000):
    """return the loss of penalized ridge regression for a fold corresponding to k_indices

    Args:
    .

    Returns:
    - loss_te (float): The testing loss.
    - f1_score (float): The F1 score for classification.
    - acc (float): The accuracy for classification.

    """

    # compute the model parameter
    w = 0
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
    y_pred = predict(w, x_test, method, decision_threshold)

    #y_pred = np.where(y_pred == 0, -1, 1)

    f1_score = calculate_f1_score(y_test, y_pred)
    acc = calculate_accuracy(y_test, y_pred)

    return acc, f1_score

def cross_validation(preprocessed_folds,  method, initial_w, gamma, lambda_, k_fold, decision_threshold):
    """
    Perform k-fold cross-validation with given parameters and evaluation function.

    Parameters:
    preprocessed_folds: A ECRIREEEEEEEEEEE
    method (str): the method name
    initial_w (numpy.ndarray): Initial model weights.
    gamma (float): The learning rate for gradient descent or other optimization methods.
    lambda_ (float): Regularization parameter.
    k_fold (int): The number of folds for cross-validation.

    Returns:
    tuple: A tuple containing mean cross-validation loss (or evaluation metric) and an array of cross-validation results.

    """
    performance = []
    for k in range(k_fold):
        x_train, y_train, x_test, y_test = preprocessed_folds[k]
        acc, f1 = cross_validation_step(x_train, y_train, x_test, y_test, method, lambda_, initial_w,  gamma,  decision_threshold, max_iters= 1000)
        #print(f"For fold {k+1}, f1-score is {f1} and accuracy is {acc}")
        performance.append([acc, f1])
    # compute the mean of the f1 score, the accuray and the test loss

    performance = np.array(performance)
    
    mean_acc, mean_f1 = performance.mean(axis=0)
    std_acc, std_f1 = performance.std(axis=0)

    return mean_acc, std_acc, mean_f1, std_f1 


def hypertuning(y, tx, gammas, lambdas, method, decision_thresholds):
    """
    Choose the best hyperparameters for a model using cross-validation.

    Parameters:
    - tx (numpy.ndarray): Feature matrix of shape (N, D).
    - y (numpy.ndarray): Label vector of shape (N,).
    - max_iters (list): List of maximum iteration values to search through.
    - gammas (list): List of learning rate values to search through.
    - lambdas (list): List of lambda (regularization strength) values to search through.
    - regulizer_orders (list): List of regularization orders to search through.
    - f (function): A function for logistic regression or other related tasks.

    Returns:
    - best_lambda (float): The best lambda value found during cross-validation.
    - best_max_iters (int): The best maximum iteration value found during cross-validation.
    - best_gamma (float): The best gamma (learning rate) found during cross-validation.
    - best_reg (int): The best regularization order found during cross-validation.
    """
    # set the seed and the k_fold
    seed = 1
    k_fold = 2

    # split data in k fold
    preprocessed_folds = final_k_folds(tx, y, k_fold, method)

    # Define lists to store the results of cross-validation
    cross_val_params = []
    max_f1 = 0
    best_lambda_ = None
    best_gamma = None
    best_threshold = None
    best_perfs = []
    # perform cross validation for all possible combinations of hyper parameters
    for lambda_ in lambdas:
            for gamma in gammas:
                for decision_threshold in decision_thresholds:
                    initial_w = np.zeros(preprocessed_folds[0][0].shape[1])
                    mean_acc, std_acc, mean_f1, std_f1  = cross_validation(preprocessed_folds, method, initial_w, gamma, lambda_, k_fold, decision_threshold)
                    #print(f" lambda = {lambda_}, gamma = {gamma}, decision_threshold = {decision_threshold}: F1 = {mean_f1:.3f} and acc = {mean_acc:.3f}")
                    cross_val_params.append([mean_acc, std_acc, mean_f1, std_f1, lambda_, gamma, decision_threshold])
                    if(mean_f1> max_f1):
                        max_f1 = mean_f1
                        best_perfs =[mean_acc, std_acc, mean_f1, std_f1]
                        best_lambda_ = lambda_
                        best_gamma = gamma
                        best_threshold = decision_threshold
                        print(f" lambda = {lambda_}, gamma = {gamma}, decision_threshold = {best_threshold}: F1 = {mean_f1:.3f} and acc = {mean_acc:.3f}")

    # save all parameters and results for further analyis
    with open(f"model_performance/{method}_performance.pkl", "wb") as file:
        pickle.dump(cross_val_params, file)

    return (best_lambda_, best_gamma, best_threshold, best_perfs)




