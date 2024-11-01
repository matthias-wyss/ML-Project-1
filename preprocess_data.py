import numpy as np
import random
from implementations import *
from cross_validation import *
from utils import *
from predict import *

def preprocess(x_train, x_test, y_train, method):
    """
    Preprocess the training and testing data for a machine learning task.

    Parameters:
    - x_train (numpy.ndarray): Training feature matrix.
    - x_test (numpy.ndarray): Testing feature matrix.
    - y_train (numpy.ndarray): Training target labels.

    Returns:
    - x_train (numpy.ndarray): Preprocessed training feature matrix.
    - x_test (numpy.ndarray): Preprocessed testing feature matrix.
    - y_train (numpy.ndarray): Preprocessed training target labels.
    """
   # Convert y_train labels from -1 and 1 to 0 and 1
    y_train = convert_labels(y_train)
    
    # Remove columns with a single unique value from both x_train and x_test
    x_train, x_test = remove_single_value_columns(x_train, x_test)
    
    # Replace all NaN by -1 
    x_train, x_test = replace_nan(x_train, x_test, replacement_value= -1)

    if(method == "logistic_regression" or method == "reg_logistic_regression"):
        # outliers removal form the training set
        x_train, y_train = remove_outliers(x_train, y_train, threshold=2, max_false_percentage=0.15)
    
    # Standardize x_train and x_test
    x_train, x_test = normalize_or_standardize(x_train, x_test, method='standardize')
    
    # Check for imbalance and balance the data if needed
    x_train, y_train = balance_data(x_train, y_train, method="random_upsampling", target_minority_ratio=0.2, noise_ratio=0.1)
    
    # Apply PCA to training data

    
    x_train, components, train_mean = perform_pca(x_train)  

    x_test_centered = x_test - train_mean  
    # Apply PCA to testing data using the same components
    x_test = np.dot(x_test_centered, components)
        
    return x_train, x_test, y_train

    
def balance_data(x_train, y_train, method="random_upsampling", target_minority_ratio=0.2, noise_ratio=0.1, seed=None):
    """
    Check for class imbalance and balance the data according to the specified method.

    Args:
        x_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        method (str): Method for balancing ('random_downsampling', 'random_upsampling', 'noise_upsampling').
        target_minority_ratio (float): Desired ratio of the minority class in the balanced dataset.
        noise_ratio (float): The ratio of noise to add during noise_upsampling (ignored for other methods).
        random_state (int or None): Random seed for reproducibility.

    Returns:
        x_train_balanced (numpy.ndarray): Balanced training feature matrix.
        y_train_balanced (numpy.ndarray): Balanced training labels.
    """
    if seed is not None:
        np.random.seed(seed)

    # Identify unique classes and their counts
    unique_classes, class_counts = np.unique(y_train, return_counts=True)

    # Ensure the dataset is binary
    if len(unique_classes) != 2:
        raise ValueError("This function is designed for binary datasets only.")

    # Determine major and minor classes based on counts
    if class_counts[0] > class_counts[1]:
        major_class, minor_class = unique_classes[0], unique_classes[1]
        major_count, minor_count = class_counts[0], class_counts[1]
    else:
        major_class, minor_class = unique_classes[1], unique_classes[0]
        major_count, minor_count = class_counts[1], class_counts[0]

    total_count = major_count + minor_count

    print(f'Original sizes:\n  - Majority class ({major_class}): {major_count} ({100*major_count/total_count:.2f}%)\n  - Minority class ({minor_class}): {minor_count} ({100*minor_count/total_count:.2f}%)')

    # Calculate the desired total size based on the target_minority_ratio
    target_minor_size = int((target_minority_ratio * major_count) / (1 - target_minority_ratio))
    total_balanced_size = major_count + target_minor_size

    # Get indices of major and minor classes
    major_indices = np.where(y_train == major_class)[0]
    minor_indices = np.where(y_train == minor_class)[0]

    # Balance the dataset using the specified method
    if method == "random_downsampling":
        # Calculate the target majority class size to achieve the target minority ratio
        target_major_size = int(minor_count * (1 - target_minority_ratio) / target_minority_ratio)

        # Downsample the majority class
        if major_count > target_major_size:
            major_indices = np.random.choice(major_indices, size=target_major_size, replace=False)

        print(
            f"Majority class downsampled:\n  - Class ({major_class}): {len(major_indices)} ({100*len(major_indices)/(len(major_indices)+len(minor_indices)):.2f}%)\n  - Class ({minor_class}): {len(minor_indices)} ({100*len(minor_indices)/(len(major_indices)+len(minor_indices)):.2f}%)"
        )

    elif method == "random_upsampling":
        # Upsample the minority class
        if target_minor_size > minor_count:
            extra_indices = np.random.choice(
                minor_indices, size=target_minor_size - minor_count, replace=True
            )
            minor_indices = np.concatenate([minor_indices, extra_indices])

            print(
                f"Minority class upsampled:\n  - Class ({major_class}): {major_count} ({100*major_count/total_balanced_size:.2f}%)\n  - Class ({minor_class}): {target_minor_size} ({100*target_minor_size/total_balanced_size:.2f}%)"
            )

    elif method == "noise_upsampling":
        # Upsample the minority class with added noise (data augmentation) 
        if target_minor_size > minor_count:
            extra_indices = np.random.choice(
                minor_indices, size=target_minor_size - minor_count, replace=True
            )
            minor_indices = np.concatenate([minor_indices, extra_indices])

            # Generate noise and add to the newly upsampled minority class data
            new_samples = x_train[extra_indices]
            for i in range(new_samples.shape[1]):  # For each feature column
                col_min = np.min(new_samples[:, i])
                col_max = np.max(new_samples[:, i])
                noise = noise_ratio * (col_max - col_min) * np.random.randn(len(new_samples))
                new_samples[:, i] += noise  # Add noise to each feature

            # Replace the extra samples with the noisy ones
            x_train[extra_indices] = new_samples
            
        print(
            f"Minority class upsampled with noise ratio {noise_ratio}:\n  - Class ({major_class}): {major_count} ({100*major_count/total_balanced_size:.2f}%)\n  - Class ({minor_class}): {target_minor_size} ({100*target_minor_size/total_balanced_size:.2f}%)"
        )

    else:
        raise ValueError(f"Method {method} not implemented. Use 'random_downsampling', 'random_upsampling', or 'noise_upsampling'.")

    # Combine indices of major and minor classes and shuffle
    balanced_indices = np.concatenate([major_indices, minor_indices])
    np.random.shuffle(balanced_indices)

    # Extract the balanced data
    x_train_balanced = x_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]

    return x_train_balanced, y_train_balanced
    

def convert_labels(y_train):
    """
    Convert the labels from -1 and 1 to 0 and 1.
    Args:
        y_train (numpy.ndarray): Training labels containing values -1 and 1.

    Returns:
        y_train_converted (numpy.ndarray): Converted labels containing values 0 and 1.
    """
    # Map -1 to 0 and keep 1 as is
    y_train_converted = np.where(y_train == -1, 0, y_train)
    
    print("Labels converted from -1 to 0.")

    return y_train_converted


def convert_labels_back(y_train):
    """
    Convert the labels from 0 and 1 to -1 and 1.

    Args:
        y_train (numpy.ndarray): Training labels containing values 0 and 1.

    Returns:
        y_train_converted (numpy.ndarray): Converted labels containing values -1 and 1.
    """
    # Map 0 to -1 and keep 1 as is
    y_train_converted = np.where(y_train == 0, -1, y_train)
    
    print("Labels converted from 0 to -1.")

    return y_train_converted
    

def remove_single_value_columns(x_train, x_test):
    """
    Remove columns from x_train and x_test where all values are the same (nunique == 1).

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.

    Returns:
        x_train_cleaned (numpy.ndarray): Updated x_train with single-value columns removed.
        x_test_cleaned (numpy.ndarray): Updated x_test with single-value columns removed.
    """
    # Shape before
    num_feature_before = x_train.shape[1]

    # Identify columns in x_train with a single unique value
    single_value_columns = [
        col
        for col in range(x_train.shape[1])
        if len(np.unique(x_train[:, col][~np.isnan(x_train[:, col])])) == 1
    ]

    # Filter the columns in x_train and x_test
    x_train_cleaned = np.delete(x_train, single_value_columns, axis=1)
    x_test_cleaned = np.delete(x_test, single_value_columns, axis=1)

    # Shape after
    num_feature_after = x_train_cleaned.shape[1]
    
    # Print the updated shape of the datasets
    print(f"{num_feature_before - num_feature_after} features where all values are the same removed.")
    
    return x_train_cleaned, x_test_cleaned
    

def replace_nan(x_train, x_test, replacement_value=-1):
    """
    Replace all NaN values in x_train and x_test with the specified replacement value.

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.
        replacement_value (float or int): Value to replace NaN with (default is -1).

    Returns:
        x_train_cleaned (numpy.ndarray): Updated x_train with NaN replaced by the replacement_value.
        x_test_cleaned (numpy.ndarray): Updated x_test with NaN replaced by the replacement_value.
    """
    # Replace NaN values with the specified replacement_value in both x_train and x_test
    x_train_cleaned = np.where(np.isnan(x_train), replacement_value, x_train)
    x_test_cleaned = np.where(np.isnan(x_test), replacement_value, x_test)

    print(f"Replaced all NaN values with {replacement_value}.")

    return x_train_cleaned, x_test_cleaned
    
    
def remove_rows_with_high_nan(x_train, y_train, threshold=0.1):
    """
    Remove rows from x_train where the percentage of NaN values exceeds the specified threshold.

    Args:
        x_train (numpy.ndarray): Training dataset.
        y_train (numpy.ndarray): Training labels.
        threshold (float): Percentage threshold for NaN values in rows.

    Returns:
        x_train_cleaned (numpy.ndarray): Updated x_train with rows removed.
        y_train_cleaned (numpy.ndarray): Updated y_train with corresponding rows removed.
    """
    # Calculate the percentage of NaN values in each row
    nan_percentage_per_row = np.mean(np.isnan(x_train), axis=1)

    # Identify rows where NaN percentage exceeds the threshold
    rows_to_keep = np.where(nan_percentage_per_row <= threshold)[0]

    # Keep only the valid rows in x_train and y_train
    x_train_cleaned = x_train[rows_to_keep]
    y_train_cleaned = y_train[rows_to_keep]

    print(f"Removed {x_train.shape[0] - len(rows_to_keep)} rows with NaN percentage higher than {threshold*100}%.")

    return x_train_cleaned, y_train_cleaned
    

def fill_missing_values(x_data, continuous_threshold=10):
    """
    Fill missing values in the dataset based on the nature of the features (continuous or categorical).

    Args:
        x_data (numpy.ndarray): Input dataset with missing values.
        continuous_threshold (int): Threshold to determine if a feature is continuous (number of unique values).

    Returns:
        x_filled (numpy.ndarray): Dataset with missing values filled.
    """
    # Create a copy of the original data to avoid modifying it directly
    x_filled = x_data.copy()
    
    # Initialize counters for filled columns
    continuous_count = 0
    categorical_count = 0

    # Iterate through each column in the dataset
    for col in range(x_data.shape[1]):
        col_data = x_data[:, col]
        # Check for NaNs in the column
        if np.any(np.isnan(col_data)):
            # Find unique values in the column, ignoring NaNs
            unique_values = np.unique(col_data[~np.isnan(col_data)])

            # Check if the column is continuous based on the number of unique values
            if len(unique_values) > continuous_threshold:
                # Fill NaNs with the mean value of the column
                mean_value = np.nanmean(col_data)
                x_filled[:, col] = np.where(np.isnan(col_data), mean_value, col_data)
                continuous_count += 1
                
            else:
                # For categorical columns, fill NaNs with the mode value
                mode_value = np.nan if len(unique_values) == 0 else unique_values[0]
                x_filled[:, col] = np.where(np.isnan(col_data), mode_value, col_data)
                categorical_count += 1

    # Print the total number of filled columns of each type 
    print(f"Total filled continuous columns: {continuous_count}")
    print(f"Total filled categorical columns: {categorical_count}")

    return x_filled
    

def normalize_or_standardize(x_train, x_test, method="standardize"):
    """
    Normalize or standardize the dataset.

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.
        method (str): Method for scaling, either 'normalize' (min-max) or 'standardize' (z-score).

    Returns:
        x_train_scaled (numpy.ndarray): Scaled training dataset.
        x_test_scaled (numpy.ndarray): Scaled testing dataset.
    """
    if method == "normalize":
        # Min-max normalization to scale features between 0 and 1
        min_vals = np.nanmin(x_train, axis=0)
        max_vals = np.nanmax(x_train, axis=0)
        x_train_scaled = (x_train - min_vals) / (max_vals - min_vals)
        x_test_scaled = (x_test - min_vals) / (max_vals - min_vals)

        print("Data normalized using min-max scaling.")
    elif method == "standardize":
        # Z-score standardization (zero mean, unit variance)
        means = np.nanmean(x_train, axis=0)
        stds = np.nanstd(x_train, axis=0)
        x_train_scaled = (x_train - means) / stds
        x_test_scaled = (x_test - means) / stds

        print("Data standardized using z-score scaling.")
    else:
        raise ValueError(f"Normalization method '{method}' is not recognized.")
        
    return x_train_scaled, x_test_scaled
    

def remove_outliers(x_train, y_train, threshold=3, max_false_percentage=0.1):
    """
    Remove outliers from the dataset based on z-score and a false percentage threshold.

    Args:
        x_train (numpy.ndarray): Training dataset.
        y_train (numpy.ndarray): Training labels.
        threshold (float): Z-score threshold to identify outliers.
        max_false_percentage (float): Maximum percentage of non-outlier values allowed in a row.

    Returns:
        x_train_cleaned (numpy.ndarray): Training dataset with outliers removed.
        y_train_cleaned (numpy.ndarray): Training labels with outliers removed.
    """
    # Compute the z-scores of x_train
    z_scores = np.abs((x_train - np.nanmean(x_train, axis=0)) / np.nanstd(x_train, axis=0))

    # Create a mask of non-outliers
    outlier_mask = z_scores < threshold

    # Calculate the percentage of non-outlier values in each row of the mask
    false_percentages = 1 - np.mean(outlier_mask, axis=1)

    # Check if the false percentage is less than the maximum allowed
    non_outlier_rows = false_percentages < max_false_percentage

    # Remove rows with outliers based on the non-outlier mask
    x_train_cleaned = x_train[non_outlier_rows]
    y_train_cleaned = y_train[non_outlier_rows]

    num_removed = np.sum(~non_outlier_rows)
    print(f"Removed {num_removed} outliers (z-score > {threshold}).")

    return x_train_cleaned, y_train_cleaned
    

def clip_outliers(x_train, threshold=3):
    """
    Clip outliers in the dataset to the z-score threshold.

    Args:
        x_train (numpy.ndarray): Training dataset.
        threshold (float): Z-score threshold to identify outliers.

    Returns:
        x_train_clipped (numpy.ndarray): Training dataset with outliers clipped.
    """
    # Compute the z-scores of x_train
    z_scores = np.abs((x_train - np.nanmean(x_train, axis=0)) / np.nanstd(x_train, axis=0))

    # Clip outliers to the threshold
    x_train_clipped = np.where(z_scores > threshold, np.sign(x_train) * threshold, x_train)

    print(f"Clipped outliers (z-score > {threshold}) to the threshold.")

    return x_train_clipped
    

def perform_pca(x, ratio=0.95):
    """
    Perform PCA on the given dataset.

    Args:
        x (numpy.ndarray): The input data (samples x features).
        ratio (float): Feature Retention Ration => Ratio of number of features after PCA / number of features before PCA. 

    Returns:
        x_pca (numpy.ndarray): Transformed data in the PCA space.
        components (numpy.ndarray): The principal components (eigenvectors).
        mean (numpy.ndarray): Mean vector used for centering.
    """
    # Get the number of features before PCA
    n_features_before = x.shape[1]

    # Calculate the number of components to keep
    n_components = int(n_features_before * ratio)

    x_mean = np.mean(x, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(x, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    top_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Transform the data
    x_pca = np.dot(x, top_eigenvectors)
    
    print(f'PCA performed to reduce features from {n_features_before} to {n_components}.')

    return x_pca, top_eigenvectors, x_mean
    

def add_bias_term(x_train, x_test):
    """
    Add a bias term (column of ones) to x_train and x_test.

    Args:
        x_train (numpy.ndarray): Training dataset.
        x_test (numpy.ndarray): Testing dataset.

    Returns:
        x_train_with_bias (numpy.ndarray): Training dataset with bias term added.
        x_test_with_bias (numpy.ndarray): Testing dataset with bias term added.
    """
    # Add a column of ones to the start of x_train and x_test
    x_train_with_bias = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    x_test_with_bias = np.hstack([np.ones((x_test.shape[0], 1)), x_test])

    print("Bias term added to feature matrices.")

    return x_train_with_bias, x_test_with_bias
    