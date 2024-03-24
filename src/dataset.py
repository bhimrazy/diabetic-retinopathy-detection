import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import v2 as T


class DRDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.image_paths, self.labels = self.load_csv_data()

    def load_csv_data(self):
        # Check if CSV file exists
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file '{self.csv_path}' not found.")

        # Load data from CSV file
        data = pd.read_csv(self.csv_path)

        # Check if 'image_path' and 'label' columns exist
        if "image_path" not in data.columns or "label" not in data.columns:
            raise ValueError("CSV file must contain 'image_path' and 'label' columns.")

        # Extract image paths and labels
        image_paths = data["image_path"].tolist()
        labels = data["label"].tolist()

        # Check if any image paths are invalid
        invalid_image_paths = [
            img_path for img_path in image_paths if not os.path.isfile(img_path)
        ]
        if invalid_image_paths:
            raise FileNotFoundError(f"Invalid image paths found: {invalid_image_paths}")

        # Convert labels to LongTensor
        labels = torch.LongTensor(labels)

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = read_image(image_path)
        except Exception as e:
            raise IOError(f"Error loading image at path '{image_path}': {e}")

        # Apply transformations if provided
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                raise RuntimeError(
                    f"Error applying transformations to image at path '{image_path}': {e}"
                )

        return image, label


class DRDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define the transformations
        self.train_transform = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transform = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.num_classes = 5

    def setup(self, stage=None):
        self.train_dataset = DRDataset("data/train.csv", transform=self.train_transform)
        self.val_dataset = DRDataset("data/val.csv", transform=self.val_transform)

        # compute class weights
        labels = self.train_dataset.labels.numpy()
        self.class_weights = self.compute_class_weights(labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def compute_class_weights(self, labels):
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)
