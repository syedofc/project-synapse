# ==============================================================================
# File 05a: 05_data/05a_cifar100_loader.py
# Purpose: To load and preprocess the CIFAR-100 dataset.
# ==============================================================================
"""
CIFAR-100 Data Loading and Preprocessing.

This script provides a function to get the CIFAR-100 train and test DataLoaders.
It handles transformations, normalization, and batching.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar100_loaders(data_path, batch_size, num_workers=4):
    """
    Creates and returns the CIFAR-100 train and test DataLoaders.

    Args:
        data_path (str): The path to store/load the CIFAR-100 data.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): The DataLoader for the training set.
            - test_loader (DataLoader): The DataLoader for the test set.
            - input_dim (tuple): The dimensions of a single input sample (C, H, W).
            - num_classes (int): The number of classes in the dataset.
    """
    # Define transformations for the training and test sets
    # Mean and std for CIFAR-100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download and load the CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # Helps speed up CPU to GPU transfers
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    input_dim = (3, 32, 32)
    num_classes = 100
    
    print(f"CIFAR-100 dataset loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, input_dim, num_classes
