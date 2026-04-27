#!/usr/bin/env python3
"""
inference.py - Inference utilities for Cell Instance Segmentation
================================================================
This module contains functions for model inference, including test-time
augmentation, tiling for large images, and prediction post-processing.

Key components:
- Inference with test-time augmentation (TTA)
- Large image handling with tiling
- Prediction refinement and post-processing
- COCO-format submission generation
"""

import json
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.ops import boxes as box_ops
from pathlib import Path

# Import functions from utils
from utils import coco_bbox_xyxy_to_xywh, stitch_predictions, refine_masks, encode_mask


def get_checkpoint_state(checkpoint):
    """Support both metadata checkpoints and legacy plain state_dict files."""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def infer_backbone_from_state_dict(state_dict):
    """
    Infer the backbone for legacy checkpoints that do not store metadata.

    ResNet variants share similar key names, so this is intentionally used only
    for the distinctive timm ConvNeXtV2 key pattern.
    """
    keys = state_dict.keys()
    if any("stages_0" in key or "stem_0" in key for key in keys):
        return "convnextv2_base"
    return None


def improved_infer(model, test_imgs, imgname2id, args):
    """
    Enhanced inference function with tiling for large images.
    
    Args:
        model: Model for inference
        test_imgs (list): List of image paths
        imgname2id (dict): Mapping of image filenames to IDs
        args: Command line arguments
        
    Returns:
        list: Predictions in COCO format
    """
    device = next(model.parameters()).device
    model.eval()
    
    records = []
    with torch.no_grad():
        for path in test_imgs:
            try:
                img_id = int(imgname2id[path.name])
                print(f"Processing {path.name} with ID {img_id}")
                
                # Read the image
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img.ndim == 2:
                    img = np.repeat(img[..., None], 3, axis=2)
                elif img.shape[2] == 4:
                    img = img[..., :3]
                
                h, w = img.shape[:2]
                orig_size = (h, w)
                
                # Decide if we need tiling
                if max(h, w) > args.max_dim:
                    print(f"  Using tiling for large image: {w}x{h}")
                    records_for_image = process_large_image_with_tiling(
                        model, img, img_id, args, device
                    )
                    records.extend(records_for_image)
                else:
                    # Standard processing for smaller images
                    records_for_image = process_standard_image(
                        model, img, img_id, args, device
                    )
                    records.extend(records_for_image)
            except KeyError as e:
                print(f"Error: Could not find ID for {path.name}, error: {e}")
                possible_matches = [k for k in imgname2id.keys() if path.name in k or k in path.name]
                if possible_matches:
                    print(f"Possible matches: {possible_matches}")
            
            # Final cleanup after each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Verify predictions
    print(f"Total predictions generated: {len(records)}")
    if len(records) == 0:
        print("WARNING: No predictions generated. Check model or thresholds.")
    
    return records


def process_large_image_with_tiling(model, img, img_id, args, device):
    """
    Process large images using tiling approach.
    
    Args:
        model: Model for inference
        img (numpy.ndarray): Input image
        img_id (int): Image ID
        args: Command line arguments
        device: Computation device
        
    Returns:
        list: Predictions for this image
    """
    from datasets import process_with_tiling
    
    h, w = img.shape[:2]
    records = []
    
    # Apply tiling with smaller tile size for large images
    tile_size = min(1024, args.max_dim)  # Use smaller tiles if max_dim is smaller
    overlap = 192     # Smaller overlap to reduce memory usage
    
    tiles, _, positions = process_with_tiling(
        img, tile_size=tile_size, overlap=overlap
    )
    
    # Process each tile with careful memory management
    tile_preds = []
    for tile_idx, (tile, pos) in enumerate(zip(tiles, positions)):
        # Convert to RGB and normalize
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile_t = TF.to_tensor(tile_rgb).to(device)
        
        # Clear cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run inference
        try:
            pred = model([tile_t])[0]
            
            # Convert masks to CPU immediately to save GPU memory
            if "masks" in pred:
                pred["masks"] = pred["masks"].cpu()
            pred["boxes"] = pred["boxes"].cpu()
            pred["labels"] = pred["labels"].cpu()
            pred["scores"] = pred["scores"].cpu()
            
            tile_preds.append(pred)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM error, process on CPU instead
                print(f"    OOM for tile {tile_idx}, processing on CPU")
                torch.cuda.empty_cache()
                with torch.cpu.device():
                    model = model.cpu()
                    pred = model([tile_t.cpu()])[0]
                    tile_preds.append(pred)
                    model = model.to(device)
            else:
                raise e
        
        # Delete tensors to free memory
        del tile_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print progress for large tile sets
        if len(tiles) > 10 and tile_idx % 5 == 0:
            print(f"    Processed tile {tile_idx+1}/{len(tiles)}")
    
    # Stitch predictions (runs on CPU)
    stitched = stitch_predictions(tile_preds, positions, img.shape)
    
    # Create records for each prediction
    for m, b, s, c in zip(
        stitched["masks"], stitched["boxes"], 
        stitched["scores"], stitched["labels"]
    ):
        if s > args.score_thresh:
            # Convert mask to binary
            if isinstance(m, np.ndarray):
                mask_binary = m.astype(np.uint8)
            else:
                mask_binary = m
            
            if np.any(mask_binary):
                records.append({
                    "image_id": img_id,
                    "bbox": coco_bbox_xyxy_to_xywh(b),
                    "score": float(s),
                    "category_id": int(c),
                    "segmentation": encode_mask(mask_binary),
                })
    
    return records


def process_standard_image(model, img, img_id, args, device):
    """
    Process standard-sized images with test-time augmentation.
    
    Args:
        model: Model for inference
        img (numpy.ndarray): Input image
        img_id (int): Image ID
        args: Command line arguments
        device: Computation device
        
    Returns:
        list: Predictions for this image
    """
    records = []
    
    # Resize image to appropriate dimensions. COCO RLE masks must be encoded
    # at the original image size, so keep the original shape for restoration.
    orig_h, orig_w = img.shape[:2]
    img, scale = resize_keep_ratio(img, args.max_dim)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = TF.to_tensor(img_rgb).to(device)
    
    # Clear cache before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test-time augmentation with memory management
    # 1. Original image
    pred_orig = model([img_t])[0]
    # Move tensors to CPU immediately
    orig_boxes = pred_orig["boxes"].cpu()
    orig_scores = pred_orig["scores"].cpu()
    orig_labels = pred_orig["labels"].cpu()
    orig_masks = pred_orig["masks"].cpu()
    del pred_orig
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Flipped image
    img_flip = torch.flip(img_t, [2])
    pred_flip = model([img_flip])[0]
    
    # Process flipped predictions and move to CPU
    boxes_flip = pred_flip["boxes"].clone().cpu()
    w = img_t.shape[2]
    boxes_flip[:, [0, 2]] = w - boxes_flip[:, [2, 0]]
    masks_flip = torch.flip(pred_flip["masks"].cpu(), [3])
    scores_flip = pred_flip["scores"].cpu()
    labels_flip = pred_flip["labels"].cpu()
    
    # Clear memory
    del pred_flip, img_flip, img_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Combine predictions on CPU to save GPU memory
    boxes = torch.cat([orig_boxes, boxes_flip])
    scores = torch.cat([orig_scores, scores_flip])
    labels = torch.cat([orig_labels, labels_flip])
    masks = torch.cat([orig_masks, masks_flip])
    
    # NMS by class
    keep_indices = []
    num_classes = 5  # background + 4 cell classes
    for cls in range(1, num_classes):
        cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
        if len(cls_indices) == 0:
            continue
            
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]
        
        keep = box_ops.nms(cls_boxes, cls_scores, args.nms_thresh)
        keep_indices.extend(cls_indices[keep].tolist())
    
    # Create records for each kept prediction
    for idx in keep_indices:
        m = masks[idx]
        b = boxes[idx]
        s = scores[idx]
        c = labels[idx]
        
        if s > args.score_thresh:
            mask_binary = (m[0] > 0.5).numpy().astype(np.uint8)
            if scale != 1.0:
                mask_binary = cv2.resize(
                    mask_binary,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_binary = mask_binary.astype(bool)
            
            if np.any(mask_binary):
                bbox = coco_bbox_xyxy_to_xywh(b / scale)
                bbox[0] = max(0.0, min(float(orig_w), bbox[0]))
                bbox[1] = max(0.0, min(float(orig_h), bbox[1]))
                bbox[2] = max(0.0, min(float(orig_w) - bbox[0], bbox[2]))
                bbox[3] = max(0.0, min(float(orig_h) - bbox[1], bbox[3]))
                records.append({
                    "image_id": img_id,
                    "bbox": bbox,
                    "score": float(s.item()),
                    "category_id": int(c.item()),
                    "segmentation": encode_mask(mask_binary),
                })
    
    return records


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


def save_predictions(records, output_file):
    """
    Save predictions to a JSON file in COCO format.
    
    Args:
        records (list): List of prediction records
        output_file (str): Path to output file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"[✓] Predictions saved to {output_file}")


def main_inference(args):
    """
    Main inference function to be called from the CLI.
    
    Args:
        args: Command line arguments
    """
    from models import CascadeMaskRCNN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = get_checkpoint_state(checkpoint)
    checkpoint_backbone = None
    if isinstance(checkpoint, dict):
        checkpoint_backbone = checkpoint.get("backbone")
    if checkpoint_backbone is None:
        checkpoint_backbone = infer_backbone_from_state_dict(state_dict)
    if checkpoint_backbone and checkpoint_backbone != args.backbone:
        print(
            f"[!] Checkpoint backbone is {checkpoint_backbone}; "
            f"overriding --backbone {args.backbone}."
        )
        args.backbone = checkpoint_backbone

    model = CascadeMaskRCNN(5, backbone=args.backbone).to(device)  # 5 = background + 4 classes
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[✓] Trainable parameters: {trainable_params / 1e6:.2f}M")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights do not match the selected backbone. "
            "Use the same --backbone that was used for training, for example "
            "`--backbone convnextv2_base` for a ConvNeXtV2 checkpoint."
        ) from exc
    model.eval()
    
    print(f"[✓] Loaded model from {args.ckpt}")
    print(f"[✓] Using device: {device}")

    data_root = Path(args.data_root)
    test_dir = data_root / "test"
    if not test_dir.exists():
        test_dir = data_root / "test_release"
    test_imgs = sorted(test_dir.glob("*.tif"))
    with open(data_root / "test_image_name_to_ids.json", encoding="utf-8") as f:
        imgname2id_data = json.load(f)
    
    # Create proper mapping
    if isinstance(imgname2id_data, list):
        imgname2id = {item['file_name']: item['id'] for item in imgname2id_data}
    else:
        imgname2id = imgname2id_data
    
    print(f"[✓] Found {len(test_imgs)} test images")
    
    records = improved_infer(model, test_imgs, imgname2id, args)
    save_predictions(records, args.out_file)
    
