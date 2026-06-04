# File: data/imagenet_loader.py
"""
ImageNet Data Loading and Preprocessing.

Adapted from user-provided code, this script provides a function to get
ImageNet train and validation DataLoaders, including transformations
and weighted sampling for the training set.
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

def get_imagenet_loaders(data_path, batch_size, num_workers=4, img_size=224, subset_size=None):
    """
    Creates and returns ImageNet train and validation DataLoaders.

    Args:
        data_path (str): The root path to the ImageNet dataset (e.g., where 'train' and 'val' folders are).
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of subprocesses for data loading.
        img_size (int): The size to resize images to (square).
        subset_size (int, optional): If provided, creates a random subset of this size for training
                                     and subset_size // 10 for validation. Defaults to None (full dataset).

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): The DataLoader for the training set.
            - val_loader (DataLoader): The DataLoader for the validation set.
            - input_dim_tuple (tuple): The dimensions of a single input image (C, H, W).
            - num_classes (int): The number of classes in the dataset.
    """
    print(f"Loading ImageNet data from {data_path}...")
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256), # Standard practice for ImageNet validation
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Ensure 'train' and 'val' directories exist
    # ImageNet standard structure has a 'val' directory for validation.
    # If your 'test' directory in CLS-LOC is the labeled validation set, use 'val' path to it.
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val') # Standard ImageNet validation folder name

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"ImageNet 'train' directory not found at: {train_dir}")
    if not os.path.isdir(val_dir):
        print(f"Warning: ImageNet 'val' directory not found at: {val_dir}. Trying 'test' directory instead for validation.")
        val_dir = os.path.join(data_path, 'test') # Fallback to user's 'test' dir if 'val' not found
        if not os.path.isdir(val_dir):
             raise FileNotFoundError(f"ImageNet 'val' or 'test' directory for validation not found at: {data_path}")

    try:
        train_dataset_full = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset_full = datasets.ImageFolder(root=val_dir, transform=val_transform)
    except Exception as e:
        print(f"Error loading ImageNet using ImageFolder. Ensure your dataset at {data_path} has 'train' and 'val' (or 'test' for val) subdirectories, each containing class-specific folders of images.")
        raise e

    num_classes = len(train_dataset_full.classes)
    if num_classes != 1000:
        print(f"Warning: Expected 1000 classes for ImageNet, but found {num_classes}. Please check dataset structure.")

    train_dataset = train_dataset_full
    val_dataset = val_dataset_full

    # Create subset if specified
    if subset_size is not None:
        val_subset_size = max(1, subset_size // 10)
        print(f"Creating subset of {subset_size} for training and {val_subset_size} for validation.")
        train_indices = torch.randperm(len(train_dataset_full))[:subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset_full))[:val_subset_size].tolist()
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)

    # Calculate weights for balanced sampling for the training set
    if isinstance(train_dataset, Subset):
        targets = [train_dataset_full.targets[i] for i in train_dataset.indices]
    else:
        targets = train_dataset_full.targets
    
    if not targets:
        print("Warning: Could not retrieve targets for weighted sampling. Using standard sampler for training.")
        train_sampler = None
        shuffle_train = True
    else:
        class_counts = torch.bincount(torch.tensor(targets), minlength=num_classes)
        # Avoid division by zero for classes not present in a small subset
        weights_per_class = 1.0 / (class_counts.float() + 1e-6) 
        samples_weights = torch.tensor([weights_per_class[t] for t in targets])
        
        train_sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True
        )
        shuffle_train = False # Sampler handles shuffling

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train, # shuffle is False if sampler is used
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"ImageNet data loaded. Num classes: {num_classes}. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    input_dim_tuple = (3, img_size, img_size)
    
    return train_loader, val_loader, input_dim_tuple, num_classes
