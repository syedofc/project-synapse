import torch
from torch.utils.data import DataLoader, Dataset


class ToyFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {"features": self.features[idx]}, self.labels[idx]


def _make_split(num_samples, centers, noise_std, seed):
    generator = torch.Generator().manual_seed(seed)
    num_classes, feature_dim = centers.shape
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    noise = torch.randn(num_samples, feature_dim, generator=generator) * noise_std
    features = centers[labels] + noise
    return features, labels


def get_toy_feature_loaders(
    batch_size,
    num_workers=0,
    feature_dim=16,
    num_classes=4,
    train_samples=256,
    val_samples=64,
    class_sep=3.0,
    noise_std=0.6,
    seed=42,
    **kwargs,
):
    center_generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, feature_dim, generator=center_generator) * class_sep

    train_features, train_labels = _make_split(
        num_samples=train_samples,
        centers=centers,
        noise_std=noise_std,
        seed=seed + 1,
    )
    val_features, val_labels = _make_split(
        num_samples=val_samples,
        centers=centers,
        noise_std=noise_std,
        seed=seed + 2,
    )

    train_dataset = ToyFeatureDataset(train_features, train_labels)
    val_dataset = ToyFeatureDataset(val_features, val_labels)

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

    return train_loader, val_loader, feature_dim, num_classes
