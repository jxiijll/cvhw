#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for training and inference.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TrainDataset, TestDataset, get_transforms
from models import create_model
from losses import FocalLoss, get_class_weights
from train import train_and_validate
from inference import inference
from utils import set_seed, get_weighted_sampler

import config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced ResNeXt Image Classification"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Command to run"
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda)"
    )
    parent_parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size"
    )
    parent_parser.add_argument(
        "--nodropout", action="store_true", help="Disable dropout"
    )
    parent_parser.add_argument(
        "--save_dir", type=str, default=config.DEFAULT_SAVE_DIR,
        help="Directory to save results"
    )

    train_parser = subparsers.add_parser(
        "train", parents=[parent_parser], help="Train the model"
    )
    train_parser.add_argument(
        "--train_data_dir", type=str, default=config.DEFAULT_TRAIN_DIR,
        help="Training data directory"
    )
    train_parser.add_argument(
        "--val_data_dir", type=str, default=config.DEFAULT_VAL_DIR,
        help="Validation data directory"
    )
    train_parser.add_argument(
        "--num_epochs", type=int, default=config.NUM_EPOCHS,
        help="Number of epochs"
    )
    train_parser.add_argument(
        "--learning_rate", type=float, default=config.LEARNING_RATE,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--criterion", type=str, default="focal",
        choices=["cross_entropy", "focal"], help="Loss function"
    )
    train_parser.add_argument(
        "--cutmix", action="store_true", help="Use CutMix augmentation"
    )
    train_parser.add_argument(
        "--weighted_loss", action="store_true", help="Use class-weighted loss"
    )
    train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    train_parser.add_argument(
        "--patience", type=int, default=config.PATIENCE,
        help="Early stopping patience"
    )

    inference_parser = subparsers.add_parser(
        "inference", parents=[parent_parser], help="Run inference"
    )
    inference_parser.add_argument(
        "--test_data_dir", type=str, default=config.DEFAULT_TEST_DIR,
        help="Test data directory"
    )
    inference_parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model weights"
    )
    inference_parser.add_argument(
        "--tta", action="store_true", help="Use Test-Time Augmentation"
    )

    return parser.parse_args()


def main():
    """Run training or inference."""
    args = parse_arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    if args.command == "train":
        set_seed(args.seed)

        train_transform = get_transforms(is_train=True)
        val_transform = get_transforms(is_train=False)

        train_dataset = TrainDataset(
            args.train_data_dir, transform=train_transform
        )
        val_dataset = TrainDataset(
            args.val_data_dir, transform=val_transform, is_valid=True
        )
        train_sampler = get_weighted_sampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        model = create_model(
            model_type="resnext101",
            num_classes=config.NUM_CLASSES,
            dropout_prob=0.0 if args.nodropout else config.DROPOUT_PROB
        )
        model = model.to(device)

        class_weights = (
            get_class_weights(train_dataset).to(device)
            if args.weighted_loss else None
        )

        if args.criterion == "focal":
            criterion = FocalLoss(
                gamma=2, alpha=class_weights, label_smoothing=0.1
            ).to(device)
            print("Using Focal Loss with gamma=2 and label smoothing=0.1")
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=0.1, weight=class_weights
            ).to(device)
            print("Using Cross Entropy Loss with label smoothing=0.1")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=config.WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=3,
            factor=0.1
        )

        train_and_validate(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler, device, args
        )

    elif args.command == "inference":
        test_transform = get_transforms(is_train=False)

        test_dataset = TestDataset(
            args.test_data_dir, transform=test_transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        model = create_model(
            model_type="resnext101",
            num_classes=config.NUM_CLASSES,
            dropout_prob=0.0 if args.nodropout else config.DROPOUT_PROB
        )

        print(f"Loading model weights from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        inference(model, test_loader, device, args)

    else:
        print("Please specify a command: train or inference")
        return


if __name__ == "__main__":
    main()
