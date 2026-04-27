#!/usr/bin/env python3
"""
utils.py - Utility functions for Cell Instance Segmentation
==========================================================
This module contains various utility functions for data processing, evaluation,
visualization, and other helper functions used across the project.

Key components:
- COCO evaluation utilities
- Bounding box conversion functions
- Training logger
- Mask refinement functions
"""

import csv
import json
import tempfile
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import skimage.io as sio
from torchvision.ops import boxes as box_ops
from pathlib import Path
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


def coco_bbox_xyxy_to_xywh(box):
    """
    Convert bounding box from [x1, y1, x2, y2] to [x, y, width, height].
    
    Args:
        box: Box coordinates in (x1, y1, x2, y2) format
        
    Returns:
        list: Box in COCO format [x, y, width, height]
    """
    # Convert tensor or numpy array to Python float scalars
    if isinstance(box, torch.Tensor):
        coords = box.tolist()
    elif isinstance(box, np.ndarray):
        coords = box.tolist()
    else:
        coords = box
        
    x0, y0, x1, y1 = coords
    # Ensure we return native Python floats
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def make_split(root, val_frac=0.1, seed=0):
    """
    Create a train/validation split of data folders.
    
    Args:
        root (Path): Root directory path
        val_frac (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility
    """
    import random
    random.seed(seed)
    folders = sorted([p.name for p in (root / "train").iterdir() if p.is_dir()])
    random.shuffle(folders)
    n_val = int(len(folders) * val_frac)
    split = {"val": folders[:n_val], "train": folders[n_val:]}
    json.dump(split, open(root / "split_info.json", "w"), indent=2)
    print(f"[✓] split_info.json created: {len(split['train'])} train / {len(split['val'])} val")


def refine_masks(masks, scores, threshold=0.5):
    """
    Refine predicted masks to improve segmentation quality.
    
    Args:
        masks (list): List of predicted masks
        scores (list): List of confidence scores
        threshold (float): Threshold for binary mask
        
    Returns:
        list: Refined masks
    """
    import cv2
    refined_masks = []
    
    for mask, score in zip(masks, scores):
        # Convert to binary based on threshold
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Ignore very small masks (likely noise)
        if np.sum(binary_mask) < 10:
            refined_masks.append(np.zeros_like(binary_mask))
            continue
        
        # Fill small holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        
        # For high confidence masks, no further refinement
        if score > 0.85:
            refined_masks.append(binary_mask)
            continue
        
        # For medium-low confidence masks, smooth edges
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        refined_masks.append(binary_mask)
    
    return refined_masks


def stitch_predictions(preds, positions, img_shape, score_threshold=0.5):
    """
    Stitch predictions from multiple tiles into a single prediction.
    
    Args:
        preds (list): List of prediction dicts for each tile
        positions (list): List of (x, y) positions for each tile
        img_shape (tuple): Shape of the original image (H, W, C)
        score_threshold (float): Confidence threshold for keeping predictions
        
    Returns:
        dict: Stitched predictions
    """
    h, w = img_shape[:2]
    stitched_boxes = []
    stitched_scores = []
    stitched_labels = []
    stitched_masks = []
    
    # Create an empty mask for the full image
    # Using int32 to avoid overflow with many mask IDs
    full_mask = np.zeros((h, w), dtype=np.int32)
    mask_id = 1
    
    # Process predictions from each tile
    for pred, (x_offset, y_offset) in zip(preds, positions):
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        masks = pred["masks"].cpu().numpy()
        
        for i, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
            if score < score_threshold:
                continue
                
            # Adjust box to original coordinates
            box_orig = box.copy()
            box_orig[0] += x_offset
            box_orig[1] += y_offset
            box_orig[2] += x_offset
            box_orig[3] += y_offset
            
            # Only include predictions whose center is in this tile
            # (to avoid duplicates in overlapping regions)
            center_x = (box_orig[0] + box_orig[2]) / 2
            center_y = (box_orig[1] + box_orig[3]) / 2
            
            tile_size = 1024  # Same size as in process_with_tiling
            
            # Check if this is the most appropriate tile for this prediction
            is_best_tile = True
            for other_pos in positions:
                if other_pos == (x_offset, y_offset):
                    continue
                    
                other_x, other_y = other_pos
                other_center_x = other_x + tile_size / 2
                other_center_y = other_y + tile_size / 2
                current_center_x = x_offset + tile_size / 2
                current_center_y = y_offset + tile_size / 2
                
                # Calculate distances to center
                current_dist = ((center_x - current_center_x)**2 + 
                               (center_y - current_center_y)**2)**0.5
                other_dist = ((center_x - other_center_x)**2 + 
                             (center_y - other_center_y)**2)**0.5
                
                if other_dist < current_dist:
                    is_best_tile = False
                    break
            
            if is_best_tile:
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, box_orig)
                
                # Limit to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue  # Invalid box
                
                # Convert binary mask
                mask_binary = (mask[0] > 0.5)
                
                # Check that mask is not empty
                if not np.any(mask_binary):
                    continue
                
                # Place mask in full image
                # Extract the relevant region of the mask (in local coordinates)
                mask_h, mask_w = mask_binary.shape
                # Calculate local coordinates within the tile
                local_y1 = max(0, y1 - y_offset)
                local_x1 = max(0, x1 - x_offset)
                local_y2 = min(mask_h, y2 - y_offset)
                local_x2 = min(mask_w, x2 - x_offset)
                
                # Ensure dimensions are valid
                if local_y2 <= local_y1 or local_x2 <= local_x1:
                    continue
                
                # Extract the relevant part of the mask
                mask_crop = mask_binary[local_y1:local_y2, local_x1:local_x2]
                
                # Place in full image
                full_region = full_mask[y1:y2, x1:x2]
                # Check dimensions match
                if full_region.shape != mask_crop.shape:
                    # Resize mask to match
                    try:
                        import cv2
                        mask_crop = cv2.resize(mask_crop.astype(np.uint8), 
                                             (full_region.shape[1], full_region.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                    except:
                        continue  # Skip if we can't resize
                
                # Update global mask
                full_mask[y1:y2, x1:x2][mask_crop > 0] = mask_id
                
                # Save data
                stitched_boxes.append(box_orig)
                stitched_scores.append(score)
                stitched_labels.append(label)
                stitched_masks.append(full_mask == mask_id)
                
                # Increment mask ID
                mask_id += 1
    
    return {
        "boxes": np.array(stitched_boxes) if stitched_boxes else np.zeros((0, 4)),
        "scores": np.array(stitched_scores) if stitched_scores else np.zeros(0),
        "labels": np.array(stitched_labels) if stitched_labels else np.zeros(0),
        "masks": stitched_masks if stitched_masks else []
    }


def evaluate_with_tta(model, dl, device, epoch, classes):
    """
    Evaluate model using Test-Time Augmentation with memory optimization.
    
    Args:
        model: Model to evaluate
        dl: DataLoader for validation data
        device: Device for computation (CPU/GPU)
        epoch: Current epoch number
        classes: List of class names
        
    Returns:
        tuple: (AP, AP50, AP75) metrics
    """
    model.eval()
    records = []
    gt_records = []
    
    with torch.no_grad():
        for i, (imgs, targs) in enumerate(dl):
            # Original image
            orig_img = imgs[0].to(device)
            
            # Process original image first
            pred_orig = model([orig_img])[0]
            orig_boxes = pred_orig["boxes"].cpu()
            orig_scores = pred_orig["scores"].cpu()
            orig_labels = pred_orig["labels"].cpu()
            orig_masks = pred_orig["masks"].cpu()
            
            # Free memory
            del pred_orig
            torch.cuda.empty_cache()
            
            # Horizontal flip (process separately to save memory)
            flipped = torch.flip(orig_img, [2])
            pred_flip = model([flipped])[0]
            
            # Transform flipped predictions back
            w = orig_img.shape[2]
            boxes_flipped = pred_flip["boxes"].clone().cpu()
            boxes_flipped[:, [0, 2]] = w - boxes_flipped[:, [2, 0]]
            masks_flipped = torch.flip(pred_flip["masks"].cpu(), [3])
            scores_flipped = pred_flip["scores"].cpu()
            labels_flipped = pred_flip["labels"].cpu()
            
            # Free memory
            del pred_flip, flipped
            torch.cuda.empty_cache()
            
            # Combine predictions without concatenating large tensors
            all_boxes = torch.cat([orig_boxes, boxes_flipped])
            all_scores = torch.cat([orig_scores, scores_flipped])
            all_labels = torch.cat([orig_labels, labels_flipped])
            
            # NMS by class (on CPU to save GPU memory)
            keep_indices = []
            num_classes = len(classes)
            for c in range(1, num_classes):
                class_indices = (all_labels == c).nonzero(as_tuple=True)[0]
                if len(class_indices) == 0:
                    continue
                
                class_boxes = all_boxes[class_indices]
                class_scores = all_scores[class_indices]
                
                # NMS on CPU
                keep = box_ops.nms(class_boxes, class_scores, 0.5)
                keep_indices.extend(class_indices[keep].tolist())
            
            # Process ground truth
            t = targs[0]
            img_id = int(t["image_id"].item())
            
            # Add GT records
            for m, b, c in zip(t["masks"], t["boxes"], t["labels"]):
                gt_records.append({
                    "id": len(gt_records) + 1,
                    "image_id": img_id,
                    "category_id": int(c.item()),
                    "segmentation": encode_mask(m.numpy()),
                    "area": float(torch.sum(m).item()),
                    "bbox": coco_bbox_xyxy_to_xywh(b.cpu()),
                    "iscrowd": 0
                })
            
            # Process each prediction individually without concatenating masks
            for idx in keep_indices:
                if idx < len(orig_masks):
                    # From orig_masks
                    m = orig_masks[idx]
                    b = orig_boxes[idx]
                    s = orig_scores[idx]
                    c = orig_labels[idx]
                else:
                    # From masks_flipped
                    adjusted_idx = idx - len(orig_masks)
                    m = masks_flipped[adjusted_idx]
                    b = boxes_flipped[adjusted_idx]
                    s = scores_flipped[adjusted_idx]
                    c = labels_flipped[adjusted_idx]
                
                if s.item() > 0.05:  # Low threshold for COCO evaluation
                    mask_binary = (m[0] > 0.5).numpy()
                    if np.sum(mask_binary) > 0:  # Check mask is not empty
                        records.append({
                            "image_id": img_id,
                            "category_id": int(c.item()),
                            "score": float(s.item()),
                            "bbox": coco_bbox_xyxy_to_xywh(b),
                            "segmentation": encode_mask(mask_binary),
                        })
            
            # Explicitly free memory
            del orig_boxes, orig_scores, orig_labels, orig_masks
            del boxes_flipped, scores_flipped, labels_flipped, masks_flipped
            del all_boxes, all_scores, all_labels
            torch.cuda.empty_cache()
            
            # Print progress
            if i % 5 == 0:
                print(f"Evaluating image {i+1}/{len(dl)}")
    
    # COCO evaluation
    image_info = [{"id": i, "height": img[0].shape[1], "width": img[0].shape[2]} 
                 for i, (img, _) in enumerate(dl)]
    
    # Create COCO for ground truth
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": image_info,
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes[1:], 1)],
        "annotations": gt_records
    }
    coco_gt.createIndex()
    
    # Save predictions temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    json.dump(records, open(tmp.name, "w"))
    
    # Load predictions
    try:
        coco_dt = coco_gt.loadRes(tmp.name)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap = float(coco_eval.stats[0])
        ap50 = float(coco_eval.stats[1])
        ap75 = float(coco_eval.stats[2])
        print(f"[E{epoch}] AP={ap:.3f}, AP50={ap50:.3f}, AP75={ap75:.3f}")
        print(f"Records GT: {len(gt_records)}, Predictions: {len(records)}")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        ap, ap50, ap75 = 0.0, 0.0, 0.0
    
    return ap, ap50, ap75


class TrainingLogger:
    """
    Logger for training metrics.
    
    This class handles logging of metrics during training, saving to CSV,
    and generating visualization plots.
    
    Args:
        log_dir (str or Path): Directory to save logs
    """
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp to identify experiment
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metrics to log
        self.train_losses = []
        self.val_metrics = []
        self.loss_components = []
        
        # CSV file for data
        self.csv_path = self.log_dir / f"training_log_{self.timestamp}.csv"
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'AP', 'AP50', 'AP75', 
                           'loss_classifier', 'loss_box_reg', 'loss_mask', 
                           'loss_objectness', 'loss_rpn_box_reg'])
    
    def log_epoch(self, epoch, train_loss, val_metrics=None, loss_components=None):
        """
        Log metrics for one epoch.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss for the epoch
            val_metrics (dict, optional): Validation metrics
            loss_components (dict, optional): Individual loss components
        """
        self.train_losses.append((epoch, train_loss))
        
        if val_metrics:
            self.val_metrics.append((epoch, val_metrics))
        
        if loss_components:
            self.loss_components.append((epoch, loss_components))
        
        # Save to CSV
        row = [epoch, train_loss]
        if val_metrics:
            row.extend([val_metrics.get('AP', 0), val_metrics.get('AP50', 0), val_metrics.get('AP75', 0)])
        else:
            row.extend([0, 0, 0])
            
        if loss_components:
            row.extend([
                loss_components.get('loss_classifier', 0),
                loss_components.get('loss_box_reg', 0),
                loss_components.get('loss_mask', 0),
                loss_components.get('loss_objectness', 0),
                loss_components.get('loss_rpn_box_reg', 0)
            ])
        else:
            row.extend([0, 0, 0, 0, 0])
            
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def plot_curves(self):
        """Generate learning curve plots."""
        # Prepare data
        epochs, train_losses = zip(*self.train_losses) if self.train_losses else ([], [])
        
        # Training loss plot
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        
        # Add validation metrics if available
        if self.val_metrics:
            val_epochs, val_data = zip(*self.val_metrics)
            ap_values = [d.get('AP', 0) for d in val_data]
            ap50_values = [d.get('AP50', 0) for d in val_data]
            ap75_values = [d.get('AP75', 0) for d in val_data]
            
            plt.plot(val_epochs, ap_values, 'g-', label='AP')
            plt.plot(val_epochs, ap50_values, 'r-', label='AP50')
            plt.plot(val_epochs, ap75_values, 'c-', label='AP75')
        
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.log_dir / f"learning_curves_{self.timestamp}.png")
        
        # Loss components plot
        if self.loss_components:
            comp_epochs, comp_data = zip(*self.loss_components)
            plt.figure(figsize=(12, 8))
            
            # Extract each component
            classifier_loss = [d.get('loss_classifier', 0) for d in comp_data]
            box_reg_loss = [d.get('loss_box_reg', 0) for d in comp_data]
            mask_loss = [d.get('loss_mask', 0) for d in comp_data]
            objectness_loss = [d.get('loss_objectness', 0) for d in comp_data]
            rpn_box_reg_loss = [d.get('loss_rpn_box_reg', 0) for d in comp_data]
            
            plt.plot(comp_epochs, classifier_loss, 'r-', label='Classifier Loss')
            plt.plot(comp_epochs, box_reg_loss, 'g-', label='Box Reg Loss')
            plt.plot(comp_epochs, mask_loss, 'b-', label='Mask Loss')
            plt.plot(comp_epochs, objectness_loss, 'c-', label='Objectness Loss')
            plt.plot(comp_epochs, rpn_box_reg_loss, 'm-', label='RPN Box Reg Loss')
            
            plt.xlabel('Epochs')
            plt.ylabel('Loss Components')
            plt.title('Loss Components Over Time')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.log_dir / f"loss_components_{self.timestamp}.png")
        
        print(f"Learning curves saved in {self.log_dir}")
    
    def save_final_report(self, best_ap, training_time, epochs_trained):
        """
        Generate final report with metrics and observations.
        
        Args:
            best_ap (float): Best AP achieved during training
            training_time (float): Total training time in minutes
            epochs_trained (int): Number of epochs trained
        """
        report_path = self.log_dir / f"training_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=== TRAINING REPORT ===\n\n")
            f.write(f"Date: {self.timestamp}\n")
            f.write(f"Epochs trained: {epochs_trained}\n")
            f.write(f"Total training time: {training_time:.2f} minutes\n")
            f.write(f"Best AP achieved: {best_ap:.4f}\n\n")
            
            # Final metrics
            if self.val_metrics:
                final_metrics = self.val_metrics[-1][1]
                f.write("Final validation metrics:\n")
                for metric, value in final_metrics.items():
                    f.write(f"- {metric}: {value:.4f}\n")
            
            # Loss analysis
            f.write("\nLoss analysis:\n")
            f.write(f"- Initial train loss: {self.train_losses[0][1]:.4f}\n")
            f.write(f"- Final train loss:   {self.train_losses[-1][1]:.4f}\n")
            
            # Observations and recommendations
            f.write("\nObservations and recommendations:\n")
            
            # Check if loss is still decreasing
            last_losses = [loss for _, loss in self.train_losses[-5:]]
            if last_losses[0] > last_losses[-1] and abs(last_losses[0] - last_losses[-1]) > 0.1:
                f.write("- Training loss is still decreasing. Consider training for more epochs.\n")
            else:
                f.write("- Training loss has plateaued; further epochs may not help much.\n")
            
            # Check validation metrics
            if self.val_metrics and len(self.val_metrics) >= 3:
                last_aps = [metrics.get('AP', 0) for _, metrics in self.val_metrics[-3:]]
                if last_aps[0] < last_aps[-1] and abs(last_aps[0] - last_aps[-1]) > 0.01:
                    f.write("- Validation AP is still improving. Consider training for more epochs.\n")
                else:
                    f.write("- Validation AP has plateaued; focus on model improvements.\n")
            
            # Analyze loss components
            if self.loss_components and len(self.loss_components) > 0:
                last_comp = self.loss_components[-1][1]
                largest_loss = max(last_comp.items(), key=lambda x: x[1])
                f.write(f"Largest loss component: {largest_loss[0]} ({largest_loss[1]:.4f})\n")
                
                if largest_loss[0] == 'loss_classifier':
                    f.write("  Suggestion: Adjust classification or class balancing.\n")
                elif largest_loss[0] == 'loss_box_reg':
                    f.write("  Suggestion: Optimize bounding box regression and anchors.\n")
                elif largest_loss[0] == 'loss_mask':
                    f.write("  Suggestion: Improve mask prediction or specific augmentations.\n")
                elif largest_loss[0] == 'loss_objectness':
                    f.write("  Suggestion: Adjust RPN for better object detection.\n")
                else:
                    f.write("  Suggestion: Optimize this component of the model.\n")
            
            f.write(f"\nCSV log file location: {self.csv_path}")
        
        print(f"Final report saved at {report_path}")


def encode_mask(binary_mask):
    """
    Encode binary mask to RLE format for COCO submission.
    
    Args:
        binary_mask (numpy.ndarray): Binary mask array
        
    Returns:
        dict: RLE encoded mask
    """
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def decode_maskobj(mask_obj):
    """
    Decode RLE format to binary mask.
    
    Args:
        mask_obj: RLE encoded mask
        
    Returns:
        numpy.ndarray: Binary mask array
    """
    return mask_utils.decode(mask_obj)

def read_maskfile(filepath):
    """
    Read mask file from disk.
    
    Args:
        filepath (str): Path to mask file
        
    Returns:
        numpy.ndarray: Mask array
    """
    mask_array = sio.imread(filepath)
    return mask_array

