#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training and validation functions.
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import torch
from torch.amp import autocast, GradScaler

from utils import (
    plot_confusion_matrix,
    plot_class_accuracy,
    plot_training_curves,
    cutmix_data
)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device,
                epoch, use_cutmix=True):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scaler = GradScaler('cuda')

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", unit="batch")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply CutMix
        cutmix_applied = False
        if use_cutmix and random.random() < 0.2:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)
            cutmix_applied = True

        # Forward pass
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(inputs)

            if cutmix_applied:
                loss = lam * criterion(outputs, targets_a) +                        (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)

        if cutmix_applied:
            correct += (lam * predicted.eq(targets_a).sum().float() +
                        (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            correct += predicted.eq(targets).sum().item()

        running_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })

    # Update scheduler if needed
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass
        else:
            scheduler.step()

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Store class-wise results
    class_correct = {}
    class_total = {}
    all_targets = []
    all_predictions = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Validation", unit="batch")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update class-wise accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                pred = predicted[i].item()

                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0

                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    class_acc = {}
    for class_idx in class_total:
        class_acc[class_idx] = 100. * class_correct[class_idx] / class_total[class_idx]

    # Print low-accuracy classes
    print("\nClasses with accuracy < 70%:")
    problem_classes = {cls: acc for cls, acc in class_acc.items() if acc < 70.0}
    for cls, acc in sorted(problem_classes.items(), key=lambda x: x[1]):
        print(f"  Class {cls}: {acc:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=range(100))

    return running_loss / len(dataloader), 100. * correct / total, class_acc, cm


def train_and_validate(model, train_loader, val_loader, criterion, optimizer,
                       scheduler, device, args):
    """
    Run the full training and validation process.
    """
    os.makedirs(args.save_dir, exist_ok=True)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Early stopping
    patience = args.patience if hasattr(args, 'patience') else 5
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{args.num_epochs} {'='*20}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, None, device,
            epoch, use_cutmix=args.cutmix
        )

        val_loss, val_acc, class_acc, cm = validate(
            model, val_loader, criterion, device, epoch
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save plots
        if (epoch + 1) % 5 == 0 or epoch == 0 or val_acc > best_val_acc:
            plot_confusion_matrix(
                cm,
                class_names=range(100),
                save_path=os.path.join(args.save_dir, f"confusion_matrix_epoch_{epoch+1}.png")
            )

            plot_class_accuracy(
                {k: v/100 for k, v in class_acc.items()},
                save_path=os.path.join(args.save_dir, f"class_accuracy_epoch_{epoch+1}.png")
            )

            history = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'train_acc': train_accs,
                'val_acc': val_accs
            }
            plot_training_curves(
                history,
                save_path=os.path.join(args.save_dir, f"training_curves_epoch_{epoch+1}.png")
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (patience: {patience})")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    np.save(os.path.join(args.save_dir, "training_history.npy"), history)

    plot_training_curves(
        history,
        save_path=os.path.join(args.save_dir, "final_training_curves.png")
    )

    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc
