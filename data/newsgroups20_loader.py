# ==============================================================================
# File 05b: 05_data/05b_20newsgroups_loader.py
# Purpose: To load and preprocess the 20 Newsgroups text dataset.
# ==============================================================================
"""
20 Newsgroups Text Data Loading and Preprocessing.

This script fetches the 20 Newsgroups dataset, converts the text to numerical
vectors using TF-IDF, and provides DataLoaders for use in PyTorch.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def get_20newsgroups_loaders(batch_size, num_workers=4, max_features=10000):
    """
    Creates and returns the 20 Newsgroups train and test DataLoaders.

    Args:
        batch_size (int): The number of samples per batch.
        num_workers (int): Subprocesses for data loading (less critical for non-image data).
        max_features (int): The maximum number of features for TF-IDF vectorization.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): The DataLoader for the training set.
            - test_loader (DataLoader): The DataLoader for the test set.
            - input_dim (int): The dimensionality of the input vectors (max_features).
            - num_classes (int): The number of classes in the dataset.
    """
    print("Fetching 20 Newsgroups dataset...")
    # Fetch the dataset, removing metadata to focus on text content
    newsgroups_train = fetch_20newsgroups(
        subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42
    )
    newsgroups_test = fetch_20newsgroups(
        subset='test', remove=('headers', 'footers', 'quotes'), shuffle=False, random_state=42
    )

    print("Vectorizing text data with TF-IDF...")
    # Initialize and fit the vectorizer on the training data
    vectorizer = TfidfVectorizer(
        max_features=max_features, stop_words='english', ngram_range=(1, 2)
    )
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    
    # Transform the test data using the same fitted vectorizer
    X_test = vectorizer.transform(newsgroups_test.data)

    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    
    # Convert sparse matrices to dense numpy arrays and then to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train.toarray()).float()
    y_train_tensor = torch.from_numpy(y_train).long()

    X_test_tensor = torch.from_numpy(X_test.toarray()).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    input_dim = max_features
    num_classes = len(newsgroups_train.target_names)

    print(f"20 Newsgroups dataset loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    return train_loader, test_loader, input_dim, num_classes
