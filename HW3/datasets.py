#!/usr/bin/env python3
"""
datasets.py - Dataset and data processing for Cell Instance Segmentation
========================================================================
This module contains dataset classes and data processing utilities for 
preparing and augmenting cell image data.

Key components:
- EnhancedCellDataset: PyTorch dataset class for cell segmentation
- Data augmentation functions tailored for cell images
- Utilities for instance mask extraction and processing
"""

import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from pathlib import Path


def resize_keep_ratio(im, max_dim):
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        im (numpy.ndarray): Input image (H, W, C) or mask (H, W)
        max_dim (int): Maximum dimension (width or height)
        
    Returns:
        tuple: (resized_image, scale_factor)
    """
    h, w = im.shape[:2]
    if max(h, w) <= max_dim:
        return im, 1.0
    scale = max_dim / max(h, w)
    new_size = (int(w * scale), int(h * scale))  # (w,h) for cv2
    interp = cv2.INTER_AREA if im.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(im, new_size, interpolation=interp), scale


def validate_boxes(boxes, masks, labels):
    """
    Validate bounding boxes to ensure positive dimensions.
    
    Args:
        boxes (list): List of bounding boxes [x0, y0, x1, y1]
        masks (list): List of corresponding masks
        labels (list): List of corresponding class labels
        
    Returns:
        tuple: (valid_boxes, valid_masks, valid_labels)
    """
    valid_indices = []
    valid_boxes = []
    valid_masks = []
    valid_labels = []
    
    for i, (box, mask, label) in enumerate(zip(boxes, masks, labels)):
        x0, y0, x1, y1 = box
        # Check for positive dimensions
        if x1 > x0 and y1 > y0:
            valid_indices.append(i)
            valid_boxes.append(box)
            valid_masks.append(mask)
            valid_labels.append(label)
        else:
            # Try to fix the box
            if x1 <= x0:
                x1 = x0 + 1
            if y1 <= y0:
                y1 = y0 + 1
            valid_indices.append(i)
            valid_boxes.append([x0, y0, x1, y1])
            valid_masks.append(mask)
            valid_labels.append(label)
    
    return valid_boxes, valid_masks, valid_labels


def mask_to_box(mask):
    """Return a torchvision-style xyxy box for a binary mask."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def advanced_augmentation(image, masks=None, labels=None):
    """
    Apply advanced augmentations to images and masks.
    
    Implements various transformations specific to cell microscopy images:
    - Geometric transformations (flips, rotations)
    - Brightness/contrast adjustments
    - Affine transformations
    - Channel-wise color adjustments
    - Gaussian noise
    - Elastic deformations for cell boundaries
    
    Args:
        image (numpy.ndarray): Input image (H, W, C)
        masks (list, optional): List of binary masks
        labels (list, optional): List of class labels
        
    Returns:
        tuple: (augmented_image, augmented_masks, labels)
    """
    # Basic geometric transformations
    flip_flag = random.random() < 0.5
    rot_k = random.randint(0, 3)
    
    # Additional augmentations
    # 1. Brightness and contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.uniform(-15, 15)    # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 2. Slight affine transformations (subtle rotations, translations)
    if random.random() < 0.3:
        h, w = image.shape[:2]
        angle = random.uniform(-10, 10)
        tx = random.uniform(-0.05, 0.05) * w
        ty = random.uniform(-0.05, 0.05) * h
        
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_REFLECT)
        
        if masks is not None:
            new_masks = []
            for mask in masks:
                m = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), 
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                new_masks.append(m > 0)
            masks = new_masks
    
    # 3. Color adjustments
    if random.random() < 0.5 and image.shape[2] == 3:
        # RGB image - apply channel-wise adjustments
        for i in range(3):
            image[:,:,i] = cv2.convertScaleAbs(image[:,:,i], 
                                              alpha=random.uniform(0.85, 1.15),
                                              beta=random.uniform(-10, 10))
    
    # 4. Gaussian noise
    if random.random() < 0.3:
        row, col, ch = image.shape
        mean = 0
        sigma = random.randint(2, 8)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        image = cv2.convertScaleAbs(image + gauss)
    
    # 5. Elastic deformations (for cell boundaries)
    if random.random() < 0.3 and masks is not None:
        h, w = image.shape[:2]
        # Create random displacement fields
        dx = np.random.rand(h, w) * 10 - 5  # Displacement range: -5 to 5 pixels
        dy = np.random.rand(h, w) * 10 - 5
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (31, 31), 5)
        dy = cv2.GaussianBlur(dy, (31, 31), 5)
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Displace the grid
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)
        
        # Apply elastic transform to image
        image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_REFLECT)
        
        # Apply to masks
        if masks is not None:
            new_masks = []
            for mask in masks:
                m = cv2.remap(mask.astype(np.uint8), map_x, map_y, 
                            interpolation=cv2.INTER_NEAREST, 
                            borderMode=cv2.BORDER_CONSTANT)
                new_masks.append(m > 0)
            masks = new_masks
    
    # Apply original transformations
    if flip_flag:
        image = cv2.flip(image, 1)
        if masks is not None:
            masks = [cv2.flip(m.astype(np.uint8), 1) > 0 for m in masks]
    
    if rot_k:
        image = np.rot90(image, k=rot_k).copy()
        if masks is not None:
            masks = [np.rot90(m, k=rot_k).copy() for m in masks]
    
    return image, masks, labels


def process_with_tiling(image, mask=None, tile_size=1024, overlap=256):
    """
    Process large images by dividing them into overlapping tiles.
    
    Args:
        image (numpy.ndarray): Input image
        mask (numpy.ndarray, optional): Input mask
        tile_size (int): Size of each tile
        overlap (int): Overlap between adjacent tiles
        
    Returns:
        tuple: (tiles, masks, positions)
    """
    h, w = image.shape[:2]
    
    # If image fits in memory, process directly
    if max(h, w) <= tile_size:
        return [image], [mask] if mask is not None else [None], [(0, 0)]
    
    tiles = []
    masks = []
    positions = []
    
    # Generate overlapping tiles
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            # Ensure we get the full size when possible
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)
            
            tile = image[y1:y2, x1:x2].copy()
            tiles.append(tile)
            positions.append((x1, y1))
            
            if mask is not None:
                masks.append(mask[y1:y2, x1:x2].copy())
    
    return tiles, masks if mask is not None else [None] * len(tiles), positions


def calculate_class_weights(folders, root, classes):
    """
    Calculate class weights based on instance frequency.
    
    Args:
        folders (list): List of data folders
        root (Path): Root directory path
        classes (list): List of class names
        
    Returns:
        list: Weights for each class
    """
    class_counts = [0] * (len(classes) - 1)  # Exclude background
    
    for folder in folders:
        for cls_idx, cls_name in enumerate(classes[1:], 0):
            mask_path = root / "train" / folder / f"{cls_name}.tif"
            if mask_path.exists():
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    # Count unique instance IDs excluding 0 (background)
                    instances = len(np.unique(mask)) - 1
                    class_counts[cls_idx] += instances
                except Exception as e:
                    print(f"Error processing {mask_path}: {e}")
    
    # Calculate inverse frequency weights
    total_instances = sum(class_counts)
    class_weights = [1.0]  # Weight for background
    for count in class_counts:
        if count > 0:
            # Higher weight for rare classes (capped at 5.0)
            weight = min(total_instances / (count * len(class_counts)), 5.0)
        else:
            weight = 1.0
        class_weights.append(weight)
    
    return class_weights


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    This function is needed because samples in the batch may have
    different numbers of instances.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        tuple: (images, targets)
    """
    return tuple(zip(*batch))


class EnhancedCellDataset(Dataset):
    """
    Enhanced dataset for cell instance segmentation.
    
    Features:
    - Instance mask extraction from class-specific mask files
    - Advanced data augmentation specific to cell images
    - Multi-scale training support
    - Class balancing based on instance frequency
    
    Args:
        root (str or Path): Root directory path
        folders (list): List of folder names to include
        max_dim (int): Maximum dimension for resizing
        aug (bool): Whether to apply augmentation
        multi_scale (bool): Whether to use multi-scale training
        min_dim (int): Minimum dimension for multi-scale training
        max_dim_range (int): Maximum dimension for multi-scale training
    """
    def __init__(self, root, folders, max_dim=1024, aug=False, 
                 multi_scale=False, min_dim=600, max_dim_range=1200):
        self.root = Path(root)
        self.folders = folders
        self.max_dim = max_dim
        self.aug = aug
        self.multi_scale = multi_scale
        self.min_dim = min_dim
        self.max_dim_range = max_dim_range
        
        # Get class names from global variable in the main script
        # We'll need to import this from another file
        self.classes = ["background", "class1", "class2", "class3", "class4"]
        
        # Calculate class weights for balancing
        self.class_weights = calculate_class_weights(folders, root, self.classes)
        print(f"Class weights calculated: {self.class_weights}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.folders)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image_tensor, target_dict)
        """
        folder = self.root / "train" / self.folders[idx]

        # --- Image ---------------------------------------------------------
        img = cv2.imread(str(folder / "image.tif"), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.shape[2] == 4:
            img = img[..., :3]

        # Use multi-scale training if enabled
        if self.multi_scale and self.aug:
            # Randomly select max_dim between min_dim and max_dim_range
            target_dim = random.randint(self.min_dim, self.max_dim_range)
        else:
            target_dim = self.max_dim
            
        # Decide whether to use tiling based on image size
        h, w = img.shape[:2]
        use_tiling = max(h, w) > target_dim and self.aug
        
        if not use_tiling:
            # Standard processing
            img, scale = resize_keep_ratio(img, target_dim)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get masks
            masks, boxes, labels = [], [], []
            for cls_idx, cls_name in enumerate(self.classes[1:], 1):
                m_path = folder / f"{cls_name}.tif"
                if not m_path.exists():
                    continue
                mask = cv2.imread(str(m_path), cv2.IMREAD_UNCHANGED)
                mask, _ = resize_keep_ratio(mask, target_dim)
                
                # Extract individual instance masks
                for iid in np.unique(mask)[1:]:
                    inst = (mask == iid).astype(np.uint8)
                    ys, xs = np.where(inst)
                    if ys.size == 0:
                        continue
                    box = mask_to_box(inst)
                    if box is None:
                        continue
                    boxes.append(box)
                    masks.append(inst)
                    labels.append(cls_idx)
            
            # Apply augmentations if enabled and masks exist
            if self.aug and masks:
                # Apply advanced augmentation
                img_aug, masks_aug, _ = advanced_augmentation(img_rgb, masks)
                img_t = TF.to_tensor(img_aug)
                
                # Recalculate boxes from augmented masks
                old_labels = labels
                boxes, masks, labels = [], [], []
                for mask, label in zip(masks_aug, old_labels):
                    box = mask_to_box(mask)
                    if box is None:
                        continue
                    boxes.append(box)
                    masks.append(mask)
                    labels.append(label)
            else:
                img_t = TF.to_tensor(img_rgb)
        else:
            # For very large images, take a random crop
            rand_x = random.randint(0, max(0, w - target_dim))
            rand_y = random.randint(0, max(0, h - target_dim))
            tile_w = min(target_dim, w - rand_x)
            tile_h = min(target_dim, h - rand_y)
            
            img_crop = img[rand_y:rand_y+tile_h, rand_x:rand_x+tile_w].copy()
            img_crop, scale = resize_keep_ratio(img_crop, target_dim)
            img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            
            # Process masks for the crop
            masks, boxes, labels = [], [], []
            for cls_idx, cls_name in enumerate(self.classes[1:], 1):
                m_path = folder / f"{cls_name}.tif"
                if not m_path.exists():
                    continue
                mask = cv2.imread(str(m_path), cv2.IMREAD_UNCHANGED)
                mask_crop = mask[rand_y:rand_y+tile_h, rand_x:rand_x+tile_w].copy()
                mask_crop, _ = resize_keep_ratio(mask_crop, target_dim)
                
                for iid in np.unique(mask_crop)[1:]:
                    inst = (mask_crop == iid).astype(np.uint8)
                    ys, xs = np.where(inst)
                    if ys.size == 0:
                        continue
                    # Only include masks with significant area in the crop
                    if len(ys) < 10:  # Minimum size filter
                        continue
                    box = mask_to_box(inst)
                    if box is None:
                        continue
                    boxes.append(box)
                    masks.append(inst)
                    labels.append(cls_idx)
            
            # Apply augmentation to the crop
            if self.aug and masks:
                img_aug, masks_aug, _ = advanced_augmentation(img_rgb, masks)
                img_t = TF.to_tensor(img_aug)
                
                # Recalculate boxes
                old_labels = labels
                boxes, masks, labels = [], [], []
                for mask, label in zip(masks_aug, old_labels):
                    box = mask_to_box(mask)
                    if box is None:
                        continue
                    boxes.append(box)
                    masks.append(mask)
                    labels.append(label)
            else:
                img_t = TF.to_tensor(img_rgb)
        
        # Prepare target dictionary
        if boxes:
            # Validate boxes before converting to tensors
            boxes, masks, labels = validate_boxes(boxes, masks, labels)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, img_t.shape[1], img_t.shape[2]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
        }
        return img_t, target
    
