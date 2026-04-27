#!/usr/bin/env python3
"""
train.py - Main training and inference script for Cell Instance Segmentation
===========================================================================
This is the main script that handles command-line arguments, training process, 
and inference execution.

Features:
- Command line interface for training and inference
- Training loop with validation and checkpointing
- Multi-GPU support via DataParallel
- Learning rate scheduling and gradient accumulation
"""
import argparse
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import math
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path

# Import our custom modules
from models import CascadeMaskRCNN
from datasets import EnhancedCellDataset, collate_fn
from utils import make_split, evaluate_with_tta, TrainingLogger
from inference import main_inference


# Global configuration
CLASSES = ["background", "class1", "class2", "class3", "class4"]
NUM_CLASSES = len(CLASSES)


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_state(model):
    """Return a plain state_dict, unwrapping DataParallel when needed."""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def get_checkpoint_state(checkpoint):
    """Support both metadata checkpoints and legacy plain state_dict files."""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def save_model_checkpoint(model, save_path, args):
    """Save model weights with enough metadata to rebuild the same backbone."""
    torch.save(
        {
            "model": get_model_state(model),
            "backbone": args.backbone,
            "classes": CLASSES,
        },
        save_path,
    )


def improved_train_loop(args):
    """
    Enhanced training loop with all improvements.
    
    Args:
        args: Command line arguments
    """
    set_random_seeds(args.seed)
    root = Path(args.data_root)
    
    if not (root / "split_info.json").exists():
        raise RuntimeError("First run --make_split ...")

    split = json.load(open(root / "split_info.json"))
    
    # Initialize logger
    log_dir = Path(args.out_dir) / "logs"
    logger = TrainingLogger(log_dir)
    
    # Create datasets with enhanced augmentation
    ds_train = EnhancedCellDataset(
        root, split["train"], args.max_dim, aug=bool(args.aug),
        multi_scale=args.multi_scale, min_dim=args.min_dim, max_dim_range=args.max_dim_range
    )
    ds_val = EnhancedCellDataset(root, split["val"], args.max_dim, aug=False)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers // 2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Create model with selected backbone
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Backbone: {args.backbone}")
    model = CascadeMaskRCNN(NUM_CLASSES, backbone=args.backbone).to(device)
    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    if trainable_params >= 200_000_000:
        print("WARNING: trainable parameters exceed the HW limit of 200M.")
    
    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    if args.ckpt and Path(args.ckpt).exists():
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(get_checkpoint_state(checkpoint))
        print(f"[✓] Loaded checkpoint {args.ckpt}")

    # Differential learning rates for backbone vs heads
    params = [
        {"params": [p for n, p in model.named_parameters() 
                  if "backbone" in n], "lr": args.lr / 10},
        {"params": [p for n, p in model.named_parameters() 
                  if "backbone" not in n], "lr": args.lr}
    ]
    
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    
    # One cycle learning rate schedule
    steps_per_epoch = max(1, math.ceil(len(dl_train) / args.accum_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() and args.amp else None

    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_ap = 0.0

    # Train with all improvements
    total_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        running_loss_components = {
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_mask': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0
        }
        optimizer.zero_grad()

        for it, (imgs, targets) in enumerate(dl_train, 1):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                # DataParallel compatibility
                loss_dict = model(imgs, targets)
                
                # Apply class weighting based on dataset weights
                if args.class_weights and hasattr(ds_train, 'class_weights'):
                    for t in targets:
                        if len(t["labels"]) > 0:
                            weights = torch.tensor(
                                [ds_train.class_weights[label.item()] for label in t["labels"]], 
                                device=device
                            )
                            weight_factor = weights.mean()
                            
                            if "loss_classifier" in loss_dict:
                                loss_dict["loss_classifier"] *= weight_factor
                
                losses = sum(loss_dict.values())
                loss = losses / args.accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if it % args.accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * args.accum_steps
            
            # Update loss components for logging
            for k, v in loss_dict.items():
                if k in running_loss_components:
                    running_loss_components[k] += v.item() / len(dl_train)

            if it % args.print_every == 0:
                avg = running_loss / it
                lr = optimizer.param_groups[0]['lr']
                mem = (
                    torch.cuda.max_memory_allocated() / 1024 ** 3
                    if torch.cuda.is_available()
                    else 0
                )
                print(f"E{epoch} [{it}/{len(dl_train)}] loss {avg:.4f} lr {lr:.6f} mem {mem:.2f} GB")
                # Print individual losses for debugging
                loss_str = ' '.join([f"{k}:{v.item():.4f}" for k,v in loss_dict.items()])
                print(f"  - Losses: {loss_str}")

        if len(dl_train) % args.accum_steps != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        epoch_loss = running_loss / len(dl_train)
        epoch_time = (time.time() - start) / 60
        print(f"=> epoch {epoch} done | avg {epoch_loss:.4f} | {epoch_time:.1f} min")
        
        # Log training losses
        logger.log_epoch(epoch, epoch_loss, loss_components=running_loss_components)

        # Evaluation with test-time augmentation
        if epoch % args.val_every == 0 or epoch == args.epochs:
            # Get model without DataParallel wrapper
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            ap, ap50, ap75 = evaluate_with_tta(eval_model, dl_val, device, epoch, CLASSES)
            val_metrics = {'AP': ap, 'AP50': ap50, 'AP75': ap75}
            logger.log_epoch(epoch, epoch_loss, val_metrics=val_metrics)
            
            if ap > best_ap:
                best_ap = ap
                save_path = Path(args.out_dir) / "best.pth"
                save_model_checkpoint(model, save_path, args)
                print(f"New best model! AP = {ap:.4f}")

        # Save checkpoint 
        save_path = Path(args.out_dir) / f"model_e{epoch}.pth"
        save_model_checkpoint(model, save_path, args)
    
    # Total training time
    total_time = (time.time() - total_start_time) / 60
    
    # Generate learning curves
    logger.plot_curves()
    
    # Generate final report
    logger.save_final_report(best_ap, total_time, args.epochs)
    
    print(f"[✓] Training finished | best AP={best_ap:.3f} | Total time: {total_time:.1f} min")


def get_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Cell Instance Segmentation Training and Inference")
    p.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    p.add_argument("--make_split", type=float, default=None, help="Create train/val split (e.g. 0.1)")
    p.add_argument("--mode", choices=["train", "infer"], default="train", help="Operation mode")
    
    # Training parameters
    p.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size")
    p.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    p.add_argument("--accum_steps", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--amp", action="store_true", help="Use mixed precision training")
    p.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs if available")
    p.add_argument("--cpu", action="store_true", help="Force CPU usage (no GPU)")
    p.add_argument("--class_weights", action="store_true", help="Apply class weighting")
    
    # Data parameters
    p.add_argument("--max_dim", type=int, default=1024, help="Maximum image dimension")
    p.add_argument("--min_dim", type=int, default=600, help="Min dimension for multi-scale training")
    p.add_argument("--max_dim_range", type=int, default=1200, help="Max dimension for multi-scale training")
    p.add_argument("--multi_scale", action="store_true", help="Enable multi-scale training")
    p.add_argument("--aug", type=int, default=1, help="Enable data augmentation")
    
    # Logging and checkpoints
    p.add_argument("--val_every", type=int, default=2, help="Validation frequency (epochs)")
    p.add_argument("--print_every", type=int, default=20, help="Print frequency (iterations)")
    p.add_argument("--out_dir", type=str, default="outputs_convnextv2", help="Output directory")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint to load")
    
    # Model parameters
    p.add_argument("--backbone", type=str, default="convnextv2_base", 
                   choices=["resnet50", "resnet101", "resnet152", "convnextv2_base"], 
                   help="Backbone architecture")
    
    # Inference parameters
    p.add_argument("--out_file", type=str, default="test-results.json", help="Output file for predictions")
    p.add_argument("--score_thresh", type=float, default=0.1, help="Score threshold for inference")
    p.add_argument("--nms_thresh", type=float, default=0.5, help="NMS IoU threshold for inference")
    
    return p.parse_args()


def main():
    """Main entry point."""
    args = get_args()

    if args.make_split is not None:
        make_split(Path(args.data_root), args.make_split)
        exit(0)

    if args.mode == "train":
        improved_train_loop(args)
    elif args.mode == "infer":
        if not args.ckpt:
            raise ValueError("--ckpt required in infer mode")
        main_inference(args)


if __name__ == "__main__":
    main()
