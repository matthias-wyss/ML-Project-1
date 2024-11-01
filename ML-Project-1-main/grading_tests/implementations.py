import numpy as np
import random

# 1. Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform linear regression using gradient descent.

    Parameters:
    y : array_like
        The target values.
    tx : array_like
        The feature matrix.
    initial_w : array_like
        Initial weights.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Step size (learning rate).

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    w = initial_w  # Initialize weights
    for n in range(max_iters):
        # Compute the gradient
        gradient = -tx.T @ (y - tx @ w) / len(y)
        # Update weights
        w = w - gamma * gradient
        
        if(n%100 == 0):
            print('At batch', n)
            
    # Calculate the loss
    loss = np.sum((y - tx @ w) ** 2) / (2 * len(y))
    return w, loss


# 2. Linear regression using stochastic gradient descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, seed=None):
    """
    Perform linear regression using stochastic gradient descent.

    Parameters:
    y : array_like
        The target values.
    tx : array_like
        The feature matrix.
    initial_w : array_like
        Initial weights.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Step size (learning rate).

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    w = initial_w  # Initialize weights
    random.seed(seed)
    N = len(y)
    for n in range(max_iters):
        # Compute the gradient for each sample
        i = random.randint(0, N-1)
        gradient = -(y[i] - tx[i] @ w) * tx[i]
        # Update weights
        w = w - gamma * gradient
    # Calculate the loss
    loss = np.sum((y - tx @ w) ** 2) / (2 * N)
    return w, loss

# 3. Least squares regression using normal equations
def least_squares(y, tx):
    """
    Perform least squares regression using normal equations.

    Parameters:
    y : array_like
        The target values.
    tx : array_like
        The feature matrix.

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    # Calculate optimal weights using the normal equation
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    # Calculate the loss
    loss = np.sum((y - tx @ w) ** 2) / (2 * len(y))
    return w, loss


# 4. Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """
    Perform ridge regression using normal equations.

    Parameters:
    y : array_like
        The target values.
    tx : array_like
        The feature matrix.
    lambda_ : float
        The regularization parameter.

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    # Regularization term
    lambda_prime  = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    # Calculate optimal weights using the regularized normal equation
    w = np.linalg.solve(tx.T @ tx + lambda_prime, tx.T @ y)
    # Calculate the loss
    loss = np.sum((y - tx @ w) ** 2) / (2 * len(y))
    return w, loss

# 5. Logistic regression using gradient descent
def sigmoid(t):
    """
    Compute the sigmoid function.

    Parameters:
    t : array_like
        Input values.

    Returns:
    array_like
        Sigmoid function values.
    """
    return 1 / (1 + np.exp(-t))


def logistic_regression(y, tx, initial_w, max_iters, gamma, seed=None):
    """
    Perform logistic regression using gradient descent.

    Parameters:
    y : array_like
        The target values (binary).
    tx : array_like
        The feature matrix.
    initial_w : array_like
        Initial weights.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Step size (learning rate).

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    w = initial_w  # Initialize weights
    random.seed(seed)
    N = len(y)
    for n in range(max_iters):
        # Compute the gradient
        gradient = tx.T @ (sigmoid(tx @ w) - y) / len(y)
        # Update weights
        w = w - gamma * gradient
    # Calculate the loss using cross-entropy witsigmoid(h numerical stability
    pred = sigmoid(tx @ w)
    # Ensure numerical stability by avoiding log(0) and log(1)
    eps = 1e-9  # Small epsilon to avoid log(0)
    pred[pred == 1] -= eps
    pred[pred == 0] += eps
    # Calculate the loss using cross-entropy
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return w, loss



# 6. Regularized logistic regression using gradient descent
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression using gradient descent.

    Parameters:
    y : array_like
        The target values (binary).
    tx : array_like
        The feature matrix.
    lambda_ : float
        Regularization parameter.
    initial_w : array_like
        Initial weights.
    max_iters : int
        Maximum number of iterations.
    gamma : float
        Step size (learning rate).

    Returns:
    w : array_like
        The final weights.
    loss : float
        The cost function value.
    """
    w = initial_w  # Initialize weights
    N = len(y)
    
    for n in range(max_iters):
        i = random.randint(0, N-1)
        # Compute the gradient with regularization term
        gradient = tx.T @ (sigmoid(tx @ w) - y) / len(y) + 2 * lambda_ * w
        # Update weights
        w -= gamma * gradient

    # Calculate the loss using cross-entropy with regularization
    pred = sigmoid(tx @ w)

    pred[pred == 1] -= 1e-9
    pred[pred == 0] += 1e-9
    
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return w, loss