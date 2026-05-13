"""
Main training script for the PromptIR model.

This script handles:
- Configuration of model parameters, training hyperparameters, and data paths.
- Dataset loading and splitting into training and validation sets.
- Model initialization (PromptIR).
- Definition of loss functions (L1, SSIM, Charbonnier, Perceptual).
- The main training loop, including:
    - Forward and backward passes.
    - Optimization steps.
    - Learning rate scheduling.
    - In-loop data augmentation (flips).
    - Logging of training and validation metrics.
    - Saving of the best and last model checkpoints.
- Optional Automatic Mixed Precision (AMP) support.
"""
import os
import sys
import time
import random
import argparse
import glob
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast  # For AMP

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

# Add project root to Python path for local imports.
# All project Python files now live beside this script.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from promptir_model import PromptIR
from dataset import ImageRestorationDataset
from metrics import calculate_psnr
from losses import CharbonnierLoss, SSIMLoss, VGGPerceptualLoss

# --- Configuration ---
# Training Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 2  # Per-process batch size when using accelerate.
NUM_EPOCHS = 145
IMAGE_SIZE = (256, 256)  # Target image size for training and validation
AMP_ENABLED = False  # Automatic Mixed Precision (disabled due to previous NaN issues)

# Model parameters (must match the architecture defined in PromptIR)
MODEL_BASE_DIM = 48  # Base dimension for model channels
MODEL_NUM_BLOCKS_PER_LEVEL = [3, 4, 4, 6]  # Num Transformer blocks per U-Net level
MODEL_NUM_REFINEMENT_BLOCKS = 4  # Num blocks in the refinement stage (bottleneck)
MODEL_NUM_PROMPT_COMPONENTS = 5  # Num of learnable spatial prompt components

# Prompt Generation (PGM) configuration per decoder stage (0=deep, 1=mid, 2=shallow)
MODEL_PG_PROMPT_DIM_MAP = {0: 256, 1: 128, 2: 64}  # Output channel dim of PGM
MODEL_PG_BASE_HW_MAP = {
    0: IMAGE_SIZE[0] // 16,  # Base H,W for prompt components at deepest PGM stage
    1: IMAGE_SIZE[0] // 8,   # Base H,W for prompt components at mid PGM stage
    2: IMAGE_SIZE[0] // 4    # Base H,W for prompt components at shallowest PGM stage
}
MODEL_BACKBONE_ATTN_HEADS = 8  # Num attention heads in backbone Transformer blocks
MODEL_PROMPT_INTERACTION_ATTN_HEADS = 8  # Num attention heads in prompt interaction blocks

# Loss Function Weights
L1_LOSS_WEIGHT = 0.7
SSIM_LOSS_WEIGHT = 0.15
CHARBONNIER_LOSS_WEIGHT = 0.05
PERCEPTUAL_LOSS_WEIGHT = 0.10

# Data Paths
TRAIN_DEGRADED_DIR = "Data/train/degraded"
TRAIN_CLEAN_DIR = "Data/train/clean"

# Model Saving Paths
MODEL_SAVE_DIR = "trained_models"
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "promptir_best.pth")
LAST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "promptir_last.pth")
TRAIN_STATE_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "promptir_train_state_last.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptIR.")
    parser.add_argument(
        "--data-root",
        default="Data",
        help="Dataset root containing train/degraded and train/clean.",
    )
    parser.add_argument(
        "--train-degraded-dir",
        default=None,
        help="Directory containing degraded training images. Overrides --data-root.",
    )
    parser.add_argument(
        "--train-clean-dir",
        default=None,
        help="Directory containing clean training images. Overrides --data-root.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from 'latest' or a checkpoint path. Loads full training state when available.",
    )
    parser.add_argument(
        "--use-degradation-classifier",
        action="store_true",
        help="Enable bottleneck rain/snow auxiliary classifier.",
    )
    parser.add_argument(
        "--lambda-cls",
        type=float,
        default=0.05,
        help="Weight for degradation classification loss.",
    )
    parser.add_argument(
        "--use-task-prompt-bank",
        action="store_true",
        help="Enable shared/rain/snow prompt banks with soft classifier routing.",
    )
    parser.add_argument(
        "--use-frequency-branch",
        action="store_true",
        help="Enable Laplacian high-frequency branch before refinement.",
    )
    return parser.parse_args()


def should_use_accelerate():
    return Accelerator is not None and (
        "LOCAL_RANK" in os.environ or
        "ACCELERATE_USE_CPU" in os.environ or
        "ACCELERATE_MIXED_PRECISION" in os.environ
    )


def get_device():
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    print(f"Available CUDA devices: 0-{device_count - 1}")
    try:
        user_input = input("Select CUDA device index for training [default: 0]: ").strip()
    except EOFError:
        user_input = ""

    gpu_index = 0 if user_input == "" else int(user_input)
    if gpu_index < 0 or gpu_index >= device_count:
        raise ValueError(f"Invalid CUDA device index {gpu_index}. Available range: 0-{device_count - 1}")

    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")


def save_model_state(model, path, accelerator=None):
    if accelerator is None:
        torch.save(model.state_dict(), path)
        return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), path)


def save_training_state(model, optimizer, scheduler, epoch, best_val_psnr, path, accelerator=None):
    state = {
        "epoch": epoch,
        "model_state_dict": (
            accelerator.unwrap_model(model).state_dict() if accelerator is not None else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_psnr": best_val_psnr,
    }
    if accelerator is None:
        torch.save(state, path)
        return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save(state, path)


def strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_model_weights(model, state_dict, strict, log):
    result = model.load_state_dict(strip_module_prefix(state_dict), strict=strict)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    if missing_keys:
        log("Missing checkpoint keys:")
        for key in missing_keys:
            log(f"  {key}")
    if unexpected_keys:
        log("Unexpected checkpoint keys:")
        for key in unexpected_keys:
            log(f"  {key}")
    return result


def find_latest_checkpoint():
    if os.path.exists(TRAIN_STATE_SAVE_PATH):
        return TRAIN_STATE_SAVE_PATH

    pattern = LAST_MODEL_SAVE_PATH.replace(".pth", "_epoch*.pth")
    candidates = []
    for path in glob.glob(pattern):
        match = re.search(r"_epoch(\d+)\.pth$", path)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def load_resume_checkpoint(resume_arg, model, optimizer, scheduler, device, log, allow_partial_model_load=False):
    if not resume_arg:
        return 0, 0.0

    checkpoint_path = find_latest_checkpoint() if resume_arg == "latest" else resume_arg
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        log(f"Warning: resume checkpoint not found: {resume_arg}")
        return 0, 0.0

    log(f"Loading resume checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        load_result = load_model_weights(
            model,
            checkpoint["model_state_dict"],
            strict=not allow_partial_model_load,
            log=log,
        )
        can_load_training_state = (
            len(load_result.missing_keys) == 0 and
            len(load_result.unexpected_keys) == 0
        )
        if can_load_training_state and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if can_load_training_state and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) if can_load_training_state else 0
        best_val_psnr = float(checkpoint.get("best_val_psnr", 0.0)) if can_load_training_state else 0.0
        if can_load_training_state:
            log(f"Resumed full training state from epoch {start_epoch}.")
        else:
            log("Loaded model weights with strict=False. Optimizer and scheduler start fresh.")
        return start_epoch, best_val_psnr

    load_model_weights(model, checkpoint, strict=not allow_partial_model_load, log=log)
    match = re.search(r"_epoch(\d+)\.pth$", checkpoint_path)
    start_epoch = int(match.group(1)) if match and not allow_partial_model_load else 0
    log(f"Loaded model weights only. Optimizer and scheduler start fresh at epoch {start_epoch}.")
    return start_epoch, 0.0


def extract_restored(model_output):
    if isinstance(model_output, dict):
        return model_output["restored"]
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


class CustomAugmentations:
    """
    A callable class for applying custom augmentations to pairs of PIL images.
    Note: This class is defined but not actively used in the final train_model,
    as augmentations are applied directly to tensors in the training loop.
    It's kept for potential future use or reference.
    """
    def __init__(self):
        self.rotations = [0, 90, 180, 270]

    def __call__(self, img1, img2):
        """
        Applies random horizontal flips, vertical flips, and 90-degree rotations.
        Args:
            img1 (PIL.Image): The first image (e.g., degraded).
            img2 (PIL.Image): The second image (e.g., clean).
        Returns:
            tuple: A tuple containing the augmented (img1, img2).
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img1 = T.functional.hflip(img1)
            img2 = T.functional.hflip(img2)
        # Random vertical flip
        if random.random() > 0.5:
            img1 = T.functional.vflip(img1)
            img2 = T.functional.vflip(img2)
        # Random 90-degree rotation (requires PIL Image)
        # angle = random.choice(self.rotations)
        # img1 = T.functional.rotate(img1, angle)
        # img2 = T.functional.rotate(img2, angle)
        return img1, img2


def train_model():
    """
    Main function to train the PromptIR model.
    """
    args = parse_args()
    accelerator = Accelerator() if should_use_accelerate() else None
    device = accelerator.device if accelerator is not None else get_device()

    def log(*args, **kwargs):
        if accelerator is None or accelerator.is_main_process:
            print(*args, **kwargs)

    log(f"Using device: {device}")
    if accelerator is not None:
        log(f"Accelerate enabled. Processes: {accelerator.num_processes}")
    log(
        "DAF flags: "
        f"classifier={args.use_degradation_classifier}, "
        f"task_prompt_bank={args.use_task_prompt_bank}, "
        f"frequency_branch={args.use_frequency_branch}, "
        f"lambda_cls={args.lambda_cls}"
    )

    if AMP_ENABLED:
        log("AMP Enabled for training.")
        scaler = GradScaler()
    
    # Helps in debugging NaN issues if they occur
    # torch.autograd.set_detect_anomaly(True) # Can be slow, use if needed

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        log(f"Created directory: {MODEL_SAVE_DIR}")

    train_degraded_dir = args.train_degraded_dir or os.path.join(args.data_root, "train", "degraded")
    train_clean_dir = args.train_clean_dir or os.path.join(args.data_root, "train", "clean")

    if not os.path.isdir(train_degraded_dir):
        log(f"Error: Training degraded directory not found: {train_degraded_dir}")
        return
    if not os.path.isdir(train_clean_dir):
        log(f"Error: Training clean directory not found: {train_clean_dir}")
        return

    log("Loading dataset for training...")
    train_full_dataset = ImageRestorationDataset(
        degraded_base_dir=train_degraded_dir,
        clean_base_dir=train_clean_dir,
        patch_size=IMAGE_SIZE[0],
        is_train=True
    )
    val_full_dataset = ImageRestorationDataset(
        degraded_base_dir=train_degraded_dir,
        clean_base_dir=train_clean_dir,
        patch_size=IMAGE_SIZE[0],
        is_train=False
    )
    
    # Splitting dataset into training and validation
    val_split_ratio = 0.1
    num_total_samples = len(train_full_dataset)
    val_size = max(1, int(val_split_ratio * num_total_samples)) if num_total_samples > 1 else 0
    train_size = num_total_samples - val_size
    
    # Ensure loaders are not empty, especially with small datasets/large batch sizes
    if val_size < BATCH_SIZE and val_size > 0 : 
        log(f"Warning: Validation size ({val_size}) is less than batch size ({BATCH_SIZE}). Adjusting.")
        # This case might still lead to issues if val_size is 0 after adjustment.
        # A more robust way is to ensure val_size is at least 1 if val_split_ratio > 0
    if train_size < BATCH_SIZE:
        log("Error: Dataset too small for train/val split with current batch size.")
        log("Consider using the full dataset for training or reducing batch size.")
        return # Exit if training set is too small
    
    train_index_subset, val_index_subset = random_split(
        range(num_total_samples), [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # For reproducible splits
    )
    train_dataset = Subset(train_full_dataset, list(train_index_subset))
    val_dataset = Subset(val_full_dataset, list(val_index_subset))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=(device.type == "cuda"), drop_last=True
    )
    log(f"Training dataset size: {len(train_dataset)}")

    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=(device.type == "cuda"), drop_last=False
        )
        log(f"Validation dataset size: {len(val_dataset)}")
    else:
        val_loader = None
        log("No validation dataset or validation set is empty.")

    log("Initializing PromptIR model...")
    model = PromptIR(
        in_channels=3, out_channels=3, base_dim=MODEL_BASE_DIM,
        num_blocks_per_level=MODEL_NUM_BLOCKS_PER_LEVEL,
        num_refinement_blocks=MODEL_NUM_REFINEMENT_BLOCKS,
        num_prompt_components=MODEL_NUM_PROMPT_COMPONENTS,
        pg_prompt_dim_map=MODEL_PG_PROMPT_DIM_MAP,
        pg_base_hw_map=MODEL_PG_BASE_HW_MAP,
        backbone_num_attn_heads=MODEL_BACKBONE_ATTN_HEADS,
        prompt_interaction_num_attn_heads=MODEL_PROMPT_INTERACTION_ATTN_HEADS,
        use_degradation_classifier=args.use_degradation_classifier,
        use_task_prompt_bank=args.use_task_prompt_bank,
        use_frequency_branch=args.use_frequency_branch,
        bias=False  # Typically False for models with normalization layers
    )

    model.to(device)

    # Optimizer and Loss Functions
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_l1 = nn.L1Loss().to(device)
    criterion_ssim = SSIMLoss(data_range=1.0, channel=3).to(device)
    criterion_char = CharbonnierLoss(eps=1e-3).to(device)
    criterion_perceptual = VGGPerceptualLoss().to(device)
    log("Losses: L1, SSIM, Charbonnier, Perceptual (VGG-based).")
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
    )

    start_epoch, best_val_psnr = load_resume_checkpoint(
        args.resume,
        model,
        optimizer,
        scheduler,
        device,
        log,
        allow_partial_model_load=(
            args.use_degradation_classifier or
            args.use_task_prompt_bank or
            args.use_frequency_branch
        ),
    )

    if accelerator is not None:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        if val_loader is not None:
            val_loader = accelerator.prepare(val_loader)

    log(f"Starting training from epoch {start_epoch + 1} to {NUM_EPOCHS}...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_start_time = time.time()
        running_losses = {
            'l1': 0.0, 'ssim': 0.0, 'char': 0.0, 
            'percep': 0.0, 'restore': 0.0, 'cls': 0.0, 'total': 0.0,
            'cls_correct': 0.0, 'cls_count': 0.0
        }
        
        for batch_idx, batch_data in enumerate(train_loader):
            degraded_imgs, clean_imgs, degradation_labels = batch_data
            degraded_imgs = degraded_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            degradation_labels = degradation_labels.to(device)
            
            # In-loop tensor augmentations
            if random.random() > 0.5:  # Horizontal Flip
                degraded_imgs = T.functional.hflip(degraded_imgs)
                clean_imgs = T.functional.hflip(clean_imgs)
            if random.random() > 0.5:  # Vertical Flip
                degraded_imgs = T.functional.vflip(degraded_imgs)
                clean_imgs = T.functional.vflip(clean_imgs)

            optimizer.zero_grad()
            
            with autocast(enabled=AMP_ENABLED):
                model_output = model(
                    degraded_imgs,
                    return_aux=args.use_degradation_classifier,
                )
                restored_imgs = extract_restored(model_output)
                
                l1_loss = criterion_l1(restored_imgs, clean_imgs)
                ssim_loss = criterion_ssim(restored_imgs, clean_imgs)
                char_loss = criterion_char(restored_imgs, clean_imgs)
                perceptual_loss = criterion_perceptual(restored_imgs, clean_imgs)
                
                restore_loss = (L1_LOSS_WEIGHT * l1_loss +
                                SSIM_LOSS_WEIGHT * ssim_loss +
                                CHARBONNIER_LOSS_WEIGHT * char_loss +
                                PERCEPTUAL_LOSS_WEIGHT * perceptual_loss)
                cls_loss = restored_imgs.new_tensor(0.0)
                degradation_logits = None
                if args.use_degradation_classifier and isinstance(model_output, dict):
                    degradation_logits = model_output.get("degradation_logits")
                    if degradation_logits is not None:
                        cls_loss = F.cross_entropy(degradation_logits, degradation_labels)

                total_loss = restore_loss + args.lambda_cls * cls_loss
            
            if AMP_ENABLED:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            elif accelerator is not None:
                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Accumulate losses for epoch average
            running_losses['l1'] += l1_loss.item()
            running_losses['ssim'] += ssim_loss.item()
            running_losses['char'] += char_loss.item()
            running_losses['percep'] += perceptual_loss.item() # .item() for scalar
            running_losses['restore'] += restore_loss.item()
            running_losses['cls'] += cls_loss.item()
            running_losses['total'] += total_loss.item()
            if degradation_logits is not None:
                preds = degradation_logits.argmax(dim=1)
                running_losses['cls_correct'] += (preds == degradation_labels).sum().item()
                running_losses['cls_count'] += degradation_labels.numel()

            if (batch_idx + 1) % 50 == 0: # Log every 50 batches
                cls_acc = 0.0
                if degradation_logits is not None:
                    cls_acc = (degradation_logits.argmax(dim=1) == degradation_labels).float().mean().item()
                log(f"E[{epoch+1}/{NUM_EPOCHS}], B[{batch_idx+1}/{len(train_loader)}], "
                    f"L1: {l1_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}, "
                    f"Char: {char_loss.item():.4f}, Pcp: {perceptual_loss.item():.4f}, "
                    f"Restore: {restore_loss.item():.4f}, Cls: {cls_loss.item():.4f}, "
                    f"Tot: {total_loss.item():.4f}, DegAcc: {cls_acc:.3f}")
        
        # Calculate and print average losses for the epoch
        num_batches = len(train_loader)
        train_cls_acc = (
            running_losses['cls_correct'] / running_losses['cls_count']
            if running_losses['cls_count'] > 0 else 0.0
        )
        log(f"--- E[{epoch+1}] Avg Train: "
            f"L1: {running_losses['l1']/num_batches:.4f}, "
            f"SSIM: {running_losses['ssim']/num_batches:.4f}, "
            f"Char: {running_losses['char']/num_batches:.4f}, "
            f"Pcp: {running_losses['percep']/num_batches:.4f}, "
            f"Restore: {running_losses['restore']/num_batches:.4f}, "
            f"Cls: {running_losses['cls']/num_batches:.4f}, "
            f"Tot: {running_losses['total']/num_batches:.4f}, "
            f"DegAcc: {train_cls_acc:.3f} ---")

        # Validation phase
        if val_loader:
            model.eval()
            val_psnr_epoch_total = 0.0
            val_l1_epoch_total = 0.0
            val_cls_correct = 0.0
            val_cls_count = 0.0
            num_val_samples_processed = 0
            
            with torch.no_grad():
                for val_degraded, val_clean, val_labels in val_loader:
                    val_degraded = val_degraded.to(device)
                    val_clean = val_clean.to(device)
                    val_labels = val_labels.to(device)
                    
                    with autocast(enabled=AMP_ENABLED):
                        val_output = model(
                            val_degraded,
                            return_aux=args.use_degradation_classifier,
                        )
                        val_restored = extract_restored(val_output)
                        val_logits = (
                            val_output.get("degradation_logits")
                            if isinstance(val_output, dict) else None
                        )
                        if val_logits is not None:
                            val_cls_correct += (val_logits.argmax(dim=1) == val_labels).sum().item()
                            val_cls_count += val_labels.numel()
                    
                    val_l1 = criterion_l1(val_restored, val_clean)
                    val_l1_epoch_total += val_l1.item() * val_degraded.size(0)

                    for i in range(val_restored.size(0)):
                        # Ensure tensors are detached, on CPU, and in correct format for PSNR
                        restored_img_psnr = val_restored[i].detach()
                        clean_img_psnr = val_clean[i].detach()
                        psnr = calculate_psnr(restored_img_psnr, clean_img_psnr)
                        val_psnr_epoch_total += psnr if psnr != float('inf') else 35.0 # Cap inf
                    num_val_samples_processed += val_degraded.size(0)

            if accelerator is not None:
                val_totals = torch.tensor(
                    [
                        val_l1_epoch_total,
                        val_psnr_epoch_total,
                        num_val_samples_processed,
                        val_cls_correct,
                        val_cls_count,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                val_totals = accelerator.reduce(val_totals, reduction="sum")
                val_l1_epoch_total = val_totals[0].item()
                val_psnr_epoch_total = val_totals[1].item()
                num_val_samples_processed = int(val_totals[2].item())
                val_cls_correct = val_totals[3].item()
                val_cls_count = val_totals[4].item()

            avg_val_l1 = val_l1_epoch_total / num_val_samples_processed if num_val_samples_processed > 0 else 0
            avg_val_psnr = val_psnr_epoch_total / num_val_samples_processed if num_val_samples_processed > 0 else 0.0
            avg_val_cls_acc = val_cls_correct / val_cls_count if val_cls_count > 0 else 0.0
            log(
                f"--- E[{epoch+1}] Val L1: {avg_val_l1:.4f}, "
                f"Val PSNR: {avg_val_psnr:.2f} dB, DegAcc: {avg_val_cls_acc:.3f} ---"
            )

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                log(f"New best val PSNR: {best_val_psnr:.2f} dB. Saving model to {BEST_MODEL_SAVE_PATH}")
                save_model_state(model, BEST_MODEL_SAVE_PATH, accelerator)
        
        # Save model checkpoint at the end of each epoch (or less frequently if desired)
        epoch_save_path = LAST_MODEL_SAVE_PATH.replace(".pth", f"_epoch{epoch+1}.pth")
        save_model_state(model, epoch_save_path, accelerator)
        
        scheduler.step() # Step the scheduler
        save_training_state(
            model, optimizer, scheduler, epoch + 1, best_val_psnr,
            TRAIN_STATE_SAVE_PATH, accelerator
        )
        log(f"Epoch {epoch+1} took {time.time() - epoch_start_time:.2f}s. Model saved to {epoch_save_path}")

    log("Training finished.")
    if val_loader:
        log(f"Best validation PSNR achieved: {best_val_psnr:.2f} dB")
    
    final_model_path = LAST_MODEL_SAVE_PATH.replace(".pth", f"_epoch{NUM_EPOCHS}.pth")
    log(f"Final model (epoch {NUM_EPOCHS}) saved to {final_model_path}")


if __name__ == "__main__":
    train_model()
