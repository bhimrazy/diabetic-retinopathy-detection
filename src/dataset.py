import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


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

