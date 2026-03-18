#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference functions for image classification.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def predict_with_tta(model, dataloader, device, num_augmentations=5):
    """
    Run inference with test-time augmentation.
    """
    model.eval()
    all_filenames = []
    all_predictions = []

    MEAN = [0.4575, 0.4705, 0.3730]
    STD = [0.1975, 0.1955, 0.2001]

    tta_transforms = [
        transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize(550),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    ]

    with torch.no_grad():
        for batch_idx, (inputs, filenames) in enumerate(
            tqdm(dataloader, desc="Predicting with TTA")
        ):
            batch_size = inputs.size(0)
            all_probs = torch.zeros((batch_size, 100), device=device)

            inputs = inputs.to(device)
            outputs = model(inputs)
            all_probs += F.softmax(outputs, dim=1)

            for i in range(batch_size):
                img = inputs[i].cpu()
                img_np = img.numpy().transpose(1, 2, 0)
                img_np = img_np * np.array(STD) + np.array(MEAN)
                img_np = np.clip(img_np, 0, 1)
                pil_img = Image.fromarray((img_np * 255).astype(np.uint8))

                for t_idx in range(
                    1, min(num_augmentations, len(tta_transforms))
                ):
                    aug_img = tta_transforms[t_idx](pil_img)
                    aug_img = aug_img.unsqueeze(0).to(device)
                    aug_output = model(aug_img)
                    all_probs[i] += F.softmax(aug_output, dim=1).squeeze(0)

            all_probs /= min(num_augmentations, len(tta_transforms))
            _, predicted = all_probs.max(1)

            all_filenames.extend(filenames)
            all_predictions.extend(predicted.cpu().numpy())

            if batch_idx == 0:
                print("\nSample predictions:")
                for i in range(min(5, len(filenames))):
                    print(
                        f"  {filenames[i]} -> Class "
                        f"{predicted[i].item()}"
                    )

    predictions_df = pd.DataFrame({
        "image_name": all_filenames,
        "pred_label": all_predictions
    })

    return predictions_df


def predict(model, dataloader, device):
    """
    Run inference without augmentation.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, image_names in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for img_name, pred_label in zip(
                image_names, predicted.cpu().numpy()
            ):
                predictions.append((img_name, pred_label))

    predictions_df = pd.DataFrame(
        predictions, columns=["image_name", "pred_label"]
    )
    return predictions_df


def inference(model, test_loader, device, args):
    """
    Generate predictions on the test set.
    """
    print(f"Loaded {len(test_loader.dataset)} test images")

    if args.tta:
        print("Generating predictions with Test-Time Augmentation...")
        predictions_df = predict_with_tta(
            model, test_loader, device, num_augmentations=5
        )
    else:
        print("Generating predictions without augmentation...")
        predictions_df = predict(model, test_loader, device)

    output_file = os.path.join(args.save_dir, "prediction.csv")
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    print("\nFirst 10 predictions:")
    print(predictions_df.head(10))

    return predictions_df
