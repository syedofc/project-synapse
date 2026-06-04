# File: data/imagenet_feature_loader.py
"""
Data loader for ImageNet pre-extracted features.
Reads .pt files containing feature tensors and label tensors saved by
scripts/extract_imagenet_features.py.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob

class PreExtractedFeatureDataset(Dataset):
    def __init__(self, features_dir):
        """
        Args:
            features_dir (str): Directory containing 'batch_xxxxx_features.pt' 
                                and 'batch_xxxxx_labels.pt' files.
        """
        self.features_dir = features_dir
        self.feature_files = sorted(glob.glob(os.path.join(features_dir, "*_features.pt")))
        self.label_files = sorted(glob.glob(os.path.join(features_dir, "*_labels.pt")))

        if len(self.feature_files) != len(self.label_files) or not self.feature_files:
            raise ValueError(f"Mismatch in feature/label file counts or no files found in {features_dir}. "
                             f"Features: {len(self.feature_files)}, Labels: {len(self.label_files)}")
        
        self.num_batches = len(self.feature_files)
        self.data_per_batch = [] # List of tuples (features_tensor, labels_tensor)
        
        print(f"Loading pre-extracted feature batches from {features_dir}...")
        for i in range(self.num_batches):
            try:
                features = torch.load(self.feature_files[i])
                labels = torch.load(self.label_files[i])
                # Each file contains a batch, so we store them and __getitem__ will index into samples
                for j in range(features.size(0)):
                    self.data_per_batch.append((features[j], labels[j]))
            except Exception as e:
                print(f"Error loading batch files {self.feature_files[i]} or {self.label_files[i]}: {e}")
                # Skip this batch or raise error
        
        if not self.data_per_batch:
            raise RuntimeError(f"No data samples loaded from feature files in {features_dir}. Ensure extraction was successful.")
            
        print(f"Successfully loaded {len(self.data_per_batch)} individual samples from {self.num_batches} batch files.")


    def __len__(self):
        return len(self.data_per_batch)

    def __getitem__(self, idx):
        # Returns a single feature vector and its label
        feature_vector, label = self.data_per_batch[idx]
        # The input processor for this will be an IdentityProcessor,
        # so the "inputs_dict" should contain this feature vector.
        inputs_dict = {'features': feature_vector}
        return inputs_dict, label

def get_imagenet_feature_loaders(data_path, batch_size, num_workers=0, **kwargs): # img_size not used
    """
    Creates DataLoaders for pre-extracted ImageNet features.

    Args:
        data_path (str): Path to the base directory where 'train' and 'val' subfolders
                         containing feature .pt files are located (e.g., 'data/imagenet_features').
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        **kwargs: Catches unused arguments like img_size.

    Returns:
        tuple: train_loader, val_loader, input_dim_info (feature_dim), num_classes
    """
    train_features_dir = os.path.join(data_path, 'train')
    val_features_dir = os.path.join(data_path, 'val')

    if not os.path.isdir(train_features_dir) or not os.path.isdir(val_features_dir):
        raise FileNotFoundError(f"Feature directories 'train' or 'val' not found under {data_path}. "
                                "Please run scripts/extract_imagenet_features.py first.")

    print(f"Creating feature loaders. Train features: {train_features_dir}, Val features: {val_features_dir}")
    train_dataset = PreExtractedFeatureDataset(features_dir=train_features_dir)
    val_dataset = PreExtractedFeatureDataset(features_dir=val_features_dir)

    if len(train_dataset) == 0:
        print("ERROR: Training dataset from features is empty.")
        return None, None, 0, 0
    if len(val_dataset) == 0:
        print("ERROR: Validation dataset from features is empty.")
        # Optionally return train_loader only if val is problematic but train is OK
        # return train_loader, None, feature_dim, num_classes

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(num_workers > 0)
    )

    # Determine feature_dim and num_classes from a sample
    # Assuming all feature vectors have the same dimension and labels are consistent
    sample_inputs_dict, sample_label = train_dataset[0]
    feature_dim = sample_inputs_dict['features'].shape[0] # Assuming features are (FeatureDim,)

    all_labels = torch.tensor([sample[1] for sample in train_dataset.data_per_batch], dtype=torch.long)

    input_dim_info = feature_dim 
    num_classes = int(all_labels.max().item() + 1)

    print(f"ImageNet feature loaders created. Feature dim: {feature_dim}. Num classes: {num_classes}.")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, input_dim_info, num_classes
