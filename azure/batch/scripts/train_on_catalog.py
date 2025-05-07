import logging
from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader
from galaxy_datasets.pytorch import galaxy_dataset, galaxy_datamodule
import numpy as np

def open_image_as_rgb(path: str) -> Image.Image:
    """Open image file and ensure it's RGB (3-channel)."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


class TrainingGalaxyDataset(galaxy_dataset.GalaxyDataset):
    def __init__(self, catalog, label_cols=None, transform=None, target_transform=None):
        super().__init__(
            catalog=catalog,
            label_cols=label_cols,
            transform=transform,
            target_transform=target_transform
        )

    def __getitem__(self, idx):
        galaxy = self.catalog.iloc[idx]
        file_path = galaxy['file_loc']

        # Load & convert to RGB
        image = open_image_as_rgb(file_path)

        label = galaxy[self.label_cols].astype(np.float32).values.squeeze()

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class TrainingGalaxyDataModule(galaxy_datamodule.GalaxyDataModule):
    def setup(self, stage: Optional[str] = None):
        # First call base logic to populate catalogs and transforms
        super().setup(stage)

        # # For both fit *and* validate stages, ensure train/val datasets exist
        if stage in (None, 'fit', 'validate'):
            if self.train_catalog is not None:
                self.train_dataset = TrainingGalaxyDataset(
                    catalog=self.train_catalog,
                    label_cols=self.label_cols,
                    transform=self.train_transform
                )
            if self.val_catalog is not None:
                self.val_dataset = TrainingGalaxyDataset(
                    catalog=self.val_catalog,
                    label_cols=self.label_cols,
                    transform=self.test_transform
                )