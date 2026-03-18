"""
Dataset classes and transforms for image classification.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TrainDataset(Dataset):
    """Dataset for training and validation."""

    def __init__(self, data_dir, transform=None, is_valid=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_valid = is_valid

        # Map folder names to numeric labels
        self.class_to_idx = {
            class_name: int(class_name)
            for class_name in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, class_name))
        }

        # Print class mapping for checking
        if not is_valid:
            print("Verifying class mapping (first 10 classes):")
            for i, (class_name, idx) in enumerate(
                sorted(self.class_to_idx.items(), key=lambda x: int(x[0]))
            ):
                if i < 10:
                    print(f"Folder '{class_name}' -> Index {idx}")

        self.image_paths = []
        self.labels = []

        # Collect image paths and labels
        for class_name, label in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                if self._is_image(img_name):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

        num_images = len(self.image_paths)
        num_classes = len(self.class_to_idx)
        print(f"Found {num_images} images in {num_classes} classes")

    def _is_image(self, filename):
        """Check whether a file is an image."""
        return filename.lower().endswith(('jpg', 'jpeg', 'png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """Return the number of samples in each class."""
        class_counts = {}
        for label in self.labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        return class_counts


class TestDataset(Dataset):
    """Dataset for test inference."""

    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Collect and sort test images
        self.image_files = [
            f for f in sorted(os.listdir(image_folder))
            if self._is_image(f)
        ]

    def _is_image(self, filename):
        """Check whether a file is an image."""
        return filename.lower().endswith(('jpg', 'jpeg', 'png'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return filename without extension
        image_name = os.path.splitext(self.image_files[idx])[0]
        return image, image_name


def get_transforms(is_train=True):
    """
    Return transforms for training or validation/test.
    """
    # Normalization values
    MEAN = [0.4575, 0.4705, 0.3730]
    STD = [0.1975, 0.1955, 0.2001]

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
