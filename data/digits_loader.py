from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


class DigitsFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {"features": self.features[idx]}, self.labels[idx]


def get_digits_loaders(
    batch_size,
    num_workers=0,
    test_size=0.2,
    random_state=42,
    normalize=True,
    **kwargs,
):
    digits = load_digits()
    features = torch.tensor(digits.data, dtype=torch.float32)
    labels = torch.tensor(digits.target, dtype=torch.long)

    if normalize:
        features = features / 16.0

    train_idx, val_idx = train_test_split(
        torch.arange(features.size(0)).numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=labels.numpy(),
        shuffle=True,
    )

    train_dataset = DigitsFeatureDataset(features[train_idx], labels[train_idx])
    val_dataset = DigitsFeatureDataset(features[val_idx], labels[val_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, features.size(1), len(digits.target_names)
