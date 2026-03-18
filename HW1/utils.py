#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for training and evaluation.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch
from torch.utils.data import WeightedRandomSampler


def set_seed(seed=42):
    """
    Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def cutmix_data(input_tensor, target, alpha=1.0):
    """
    Apply CutMix to a batch.
    """
    indices = torch.randperm(input_tensor.size(0))
    shuffled_input = input_tensor[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)

    # Get box coordinates
    bbx1, bby1, bbx2, bby2 = rand_bbox(input_tensor.size(), lam)

    # Replace image region
    input_tensor[:, :, bbx1:bbx2, bby1:bby2] = shuffled_input[
        :, :, bbx1:bbx2, bby1:bby2]

    # Recompute lambda by area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
               (input_tensor.size(-1) * input_tensor.size(-2)))

    return input_tensor, target, shuffled_target, lam


def rand_bbox(size, lam):
    """
    Generate a random CutMix box.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Sample box center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def plot_confusion_matrix(cm, class_names, save_path, figsize=(20, 20),
                          normalize=True):
    """
    Save a confusion matrix plot.
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=15)
    plt.colorbar()

    # Draw class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    # Draw text labels
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0.2:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=6)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_accuracy(class_accuracies, save_path, figsize=(20, 10)):
    """
    Save a per-class accuracy plot.
    """
    plt.figure(figsize=figsize)

    classes = sorted(class_accuracies.keys())
    accuracies = [class_accuracies[c] for c in classes]

    # Color low-accuracy classes in red
    colors = ['red' if acc < 0.7 else 'green' for acc in accuracies]

    bars = plt.bar(classes, accuracies, color=colors)
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7,
                label='70% Threshold')

    plt.xlabel('Class Index')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    # Draw accuracy labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=90, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_path, figsize=(20, 10)):
    """
    Save training and validation curves.
    """
    plt.figure(figsize=figsize)

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_weighted_sampler(dataset):
    """
    Create a weighted sampler.
    """
    class_counts = dataset.get_class_distribution()
    count_list = [class_counts.get(i, 0) for i in range(max(class_counts.keys()) + 1)]

    # Use inverse-frequency weights
    weights = 1.0 / torch.Tensor(count_list)
    weights = torch.nan_to_num(weights, nan=0.0)

    # Map weights to each sample
    sample_weights = weights[dataset.labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler
