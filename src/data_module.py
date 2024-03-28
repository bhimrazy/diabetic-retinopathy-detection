import lightning as L
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.dataset import DRDataset


class DRDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        val_csv_path,
        image_size: int = 224,
        batch_size: int = 8,
        num_workers: int = 4,
        use_class_weighting: bool = False,
        use_weighted_sampler: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Ensure mutual exclusivity between use_class_weighting and use_weighted_sampler
        if use_class_weighting and use_weighted_sampler:
            raise ValueError(
                "use_class_weighting and use_weighted_sampler cannot both be True"
            )

        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.use_class_weighting = use_class_weighting
        self.use_weighted_sampler = use_weighted_sampler

        # Define the transformations
        self.train_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.RandomAffine(degrees=10, translate=(0.01, 0.01), scale=(0.99, 1.01)),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.01),
                T.RandomHorizontalFlip(p=0.5),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage=None):
        """Set up datasets for training and validation."""
        # Initialize datasets with specified transformations
        self.train_dataset = DRDataset(
            self.train_csv_path, transform=self.train_transform
        )
        self.val_dataset = DRDataset(self.val_csv_path, transform=self.val_transform)

        # Compute number of classes and class weights
        labels = self.train_dataset.labels.numpy()
        self.num_classes = len(np.unique(labels))
        self.class_weights = (
            self._compute_class_weights(labels) if self.use_class_weighting else None
        )

    def train_dataloader(self):
        """Returns a DataLoader for training data."""
        if self.use_weighted_sampler:
            sampler = self._get_weighted_sampler(self.train_dataset.labels.numpy())
            shuffle = False  # Sampler will handle shuffling
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def _compute_class_weights(self, labels):
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)

    def _get_weighted_sampler(self, labels: np.ndarray) -> WeightedRandomSampler:
        """Returns a WeightedRandomSampler based on class weights.

        The weights tensor should contain a weight for each sample, not the class weights.
        Have a look at this post for an example: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
        https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
        """

        class_sample_count = np.array(
            [len(np.where(labels == label)[0]) for label in np.unique(labels)]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[label] for label in labels])
        samples_weight = torch.from_numpy(samples_weight)

        return WeightedRandomSampler(
            weights=samples_weight, num_samples=len(labels), replacement=True
        )
