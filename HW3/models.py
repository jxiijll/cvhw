#!/usr/bin/env python3
"""
models.py - Model architecture definitions for Cell Instance Segmentation
=======================================================================
This module contains the implementation of the Cascade Mask R-CNN architecture
and custom loss functions used for cell segmentation tasks.

Key components:
- CascadeMaskRCNN class: Multi-stage instance segmentation model
- DiceLoss: Loss function optimizing segmentation overlap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


# Global configuration
NUM_CLASSES = 5  # background + 4 cell classes


class TimmBackboneWithFPN(nn.Module):
    """
    Wrap a timm ImageNet-pretrained feature backbone with torchvision FPN.

    The returned features match torchvision detection models: an OrderedDict
    of multi-scale tensors and an ``out_channels`` attribute.
    """
    def __init__(
        self,
        model_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
        out_channels=256,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "The convnextv2_base backbone requires timm. "
                "Install it with `pip install timm` or `pip install -r requirements.txt`."
            ) from exc

        self.body = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        in_channels_list = self.body.feature_info.channels()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        features = self.body(x)
        features = OrderedDict((str(i), feat) for i, feat in enumerate(features))
        return self.fpn(features)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation masks.
    
    This loss function directly optimizes the overlap between predicted 
    and ground truth masks, which is closely aligned with IoU metrics used
    in evaluation. It inherently handles class imbalance between foreground 
    and background pixels.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Calculate Dice Loss between predicted masks and target masks.
        
        Args:
            pred: (N, 1, H, W) predicted mask logits
            target: (N, H, W) ground truth binary masks
            
        Returns:
            dice_loss: Scalar tensor with the dice loss value
        """
        pred = torch.sigmoid(pred)  # Convert logits to probabilities
        
        # Flatten the tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1).float()
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Dice loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CascadeMaskRCNN(nn.Module):
    """
    Cascade Mask R-CNN with multiple detection heads at different IoU thresholds.
    
    This advanced architecture addresses the mismatch between IoU thresholds used
    for training and evaluation. It employs a sequence of detectors trained with 
    increasing IoU thresholds, where each stage refines the predictions of the
    previous stage.
    
    Key improvements:
    - Progressive refinement of object detections
    - Multi-stage box regression with increasing quality requirements
    - Better handling of challenging cases (overlapping cells, irregular shapes)
    
    Args:
        num_classes (int): Number of classes (including background)
        backbone (str): Backbone architecture ('resnet50', 'resnet101',
            'resnet152', or 'convnextv2_base')
    """
    def __init__(self, num_classes, backbone="convnextv2_base"):
        super(CascadeMaskRCNN, self).__init__()
        
        # Initialize base model based on specified backbone
        if backbone == "resnet50":
            self.base_model = maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
                box_detections_per_img=2000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=2000,
                trainable_backbone_layers=5,
            )
        elif backbone == "resnet101":
            # Create ResNet101 backbone with FPN
            backbone_model = resnet_fpn_backbone(
                backbone_name='resnet101',
                weights=ResNet101_Weights.IMAGENET1K_V2,
                trainable_layers=5
            )
            
            # Create Mask R-CNN model with the custom backbone
            self.base_model = MaskRCNN(
                backbone_model,
                num_classes=num_classes,
                box_detections_per_img=2000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=2000
            )
        
        elif backbone == "resnet152":
            # Create ResNet152 backbone with FPN
            backbone_model = resnet_fpn_backbone(
                backbone_name='resnet152',
                weights=ResNet152_Weights.IMAGENET1K_V2,
                trainable_layers=5
            )
            
            # Create Mask R-CNN model with the custom backbone
            self.base_model = MaskRCNN(
                backbone_model,
                num_classes=num_classes,
                box_detections_per_img=2000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=2000
            )
        elif backbone == "convnextv2_base":
            backbone_model = TimmBackboneWithFPN()
            self.base_model = MaskRCNN(
                backbone_model,
                num_classes=num_classes,
                box_detections_per_img=2000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=2000
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify the anchor generator for better cell detection
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.base_model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, 
            aspect_ratios=aspect_ratios
        )
        
        # Increase NMS threshold for better recall
        self.base_model.rpn.nms_thresh = 0.7
        
        # Shared feature extractor and ROI heads
        self.backbone = self.base_model.backbone
        self.rpn = self.base_model.rpn
        self.roi_heads = self.base_model.roi_heads
        self.transform = self.base_model.transform
        
        # Use a single detection head. The previous multi-stage cascade logic
        # reused the same proposals during training, which created a train/infer
        # mismatch and hurt optimization.
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor with enhanced version
        in_channels = self.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)
        
        # Set min size for handling smaller images
        self.transform.min_size = (512,)
        self.transform.max_size = 1333
        
        # Replace the original predictor with ours
        self.roi_heads.box_predictor = self.box_predictor
        self.roi_heads.mask_predictor = self.mask_predictor
        
        # Dice Loss for mask refinement
        self.dice_loss = DiceLoss()
        
    def forward(self, images, targets=None):
        """
        Forward pass with cascade refinement.
        
        Args:
            images (List[Tensor]): Input images
            targets (List[Dict], optional): Ground truth boxes, labels and masks
            
        Returns:
            During training: Dict[str, Tensor] containing the losses
            During inference: List[Dict[str, Tensor]] with detection results
        """
        return self.base_model(images, targets)
