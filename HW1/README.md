# Enhanced ResNeXt with GeM Pooling for 100-Class Image Classification

## NYCU Visual Recognition using Deep Learning 2025 HW1
**Student ID:** 414551008  
**Name:** 鄧浩培  

## Overview

This project implements a 100-class image classification system for the NYCU Visual Recognition homework. The final model is built on a pretrained **ResNeXt101_32x8d** backbone and further enhanced with **GeM pooling**, **channel attention**, and **robust data augmentation** to improve generalization under strong background interference.

A key challenge of this task is that many images contain distracting background cues, such as human hands or irrelevant scene content, while some classes are also visually similar to each other. To address this issue, the model is designed to focus more on discriminative object-related features instead of overfitting to background patterns.

### Main components
- **ResNeXt101_32x8d** backbone for strong feature extraction
- **GeM pooling** for more flexible feature aggregation
- **Channel attention module** to recalibrate important feature channels
- **CutMix** augmentation for regularization
- **Focal Loss with label smoothing** for more robust optimization
- **Test-Time Augmentation (TTA)** for inference

---

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

### Directory Structure

```
.
├── config.py         # Shared configuration and hyperparameters
├── dataset.py        # Dataset classes and image transformations
├── losses.py         # Focal loss and class weighting utilities
├── main.py           # Training / inference entry point
├── models.py         # Model architecture definitions
├── train.py          # Training and validation logic
├── inference.py      # Inference and test-time augmentation
├── utils.py          # Visualization and helper functions
├── requirements.txt  # Python dependencies
└── data/
    ├── train/        # Training set (100 classes)
    ├── val/          # Validation set
    └── test/         # Test set
```

## Usage

### Training

```bash
python main.py train --train_data_dir data/train --val_data_dir data/val --save_dir ./results --cutmix --weighted_loss
```

### Inference

```bash
python main.py inference --test_data_dir data/test --model_path ./results/best_model.pth --save_dir ./results --tta
```

## Method

### 1. Data Preprocessing

To preserve fine-grained visual details, all input images are processed at a resolution of **512 × 512**. During training, we apply several augmentation techniques to improve robustness against background bias and appearance variation, including:

- `RandomResizedCrop(512, scale=(0.4, 1.0))`
- random horizontal flipping
- color jitter
- random rotation

These augmentations help reduce overfitting to fixed background patterns and encourage the model to focus on object-related features instead of incidental context.

In addition, **CutMix** is optionally applied during training as a stronger regularization strategy. By mixing image regions across samples, CutMix helps reduce reliance on local background cues and improves generalization.

### 2. Model Architecture

The proposed model is built on a pretrained **ResNeXt101_32x8d** backbone for feature extraction. On top of the backbone, we introduce two main modifications to improve classification performance under strong background interference.

First, we replace standard global average pooling with **Generalized Mean (GeM) pooling**, which provides more flexible feature aggregation and better preserves discriminative local responses.

Second, we apply a **channel attention module** inspired by squeeze-and-excitation design. This module recalibrates feature importance across channels, allowing the network to emphasize informative features while suppressing less useful background-related signals.

The final classifier consists of a dropout layer followed by a fully connected layer for 100-class prediction.

### 3. Training Details

The model is trained using **AdamW** optimizer with a learning rate of **1e-4** and weight decay of **1e-4**. Training runs for up to **40 epochs** with **early stopping** to avoid unnecessary overfitting.

For optimization, we use **Focal Loss** together with **label smoothing** to improve robustness on difficult samples and reduce overconfident predictions. We also adopt **mixed precision training** to improve computational efficiency and reduce memory usage.

