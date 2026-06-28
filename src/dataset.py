from pathlib import Path
from typing import Dict, Optional

import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.shared.config import (
    TRAIN_PROCESSED_DIR,
    VAL_PROCESSED_DIR,
    BATCH_SIZE,
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):
    """
    Custom dataset for face recognition.

    Directory structure:
        root/
            identity_1/
                img1.jpg
                img2.jpg
            identity_2/
                img1.jpg
                ...
    """

    def __init__(
        self,
        root_dir: Path,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []

        class_folders = sorted([
            folder for folder in self.root_dir.iterdir()
            if folder.is_dir()
        ])

        for folder in class_folders:
            class_name = folder.name

            if class_name not in self.class_to_idx:
                continue

            label = self.class_to_idx[class_name]

            image_paths = sorted([
                p for p in folder.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ])

            for img_path in image_paths:
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# HELPERS
# ============================================================

def build_class_mapping():
    """
    Create consistent class-to-index mapping using train identities.
    """
    all_classes = sorted([
        folder.name
        for folder in TRAIN_PROCESSED_DIR.iterdir()
        if folder.is_dir()
    ])

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(all_classes)
    }

    return class_to_idx


def create_datasets():
    """
    Create train and validation datasets.
    """
    class_to_idx = build_class_mapping()

    train_dataset = FaceDataset(
        TRAIN_PROCESSED_DIR,
        class_to_idx,
        train_transform
    )

    val_dataset = FaceDataset(
        VAL_PROCESSED_DIR,
        class_to_idx,
        val_transform
    )

    return train_dataset, val_dataset, class_to_idx


def create_dataloaders():
    """
    Create train and validation dataloaders.
    """
    train_dataset, val_dataset, class_to_idx = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, class_to_idx