"""
Defines various loss functions commonly used in image restoration tasks.

Includes:
- DirectionalDecoupledLoss: Encourages diversity among a set of prompt vectors.
- VGGPerceptualLoss: Computes perceptual loss based on VGG19 features.
- CharbonnierLoss: A robust L1-like loss.
- SSIMLoss: Structural Similarity Index Measure loss.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DirectionalDecoupledLoss(nn.Module):
    """
    Directional Decoupled Loss (DDL).

    This loss encourages a set of degradation-aware prompt vectors to be
    directionally decoupled (i.e., point in different directions).
    It penalizes pairs of prompts whose angle is less than a specified threshold.
    """
    def __init__(self, threshold_angle_degrees=90.0, epsilon=1e-8):
        """
        Args:
            threshold_angle_degrees (float): The minimum desired angle (in degrees)
                                             between pairs of prompt vectors.
            epsilon (float): A small value for numerical stability in cosine similarity
                             and acos calculations.
        """
        super().__init__()
        if not (0 <= threshold_angle_degrees <= 180):
            raise ValueError("Threshold angle must be between 0 and 180 degrees.")
        self._threshold_angle_rad = math.radians(threshold_angle_degrees)
        self.epsilon = epsilon

    def set_threshold_angle_degrees(self, new_angle_degrees):
        """Allows dynamic adjustment of the threshold angle."""
        if not (0 <= new_angle_degrees <= 180):
            raise ValueError("New threshold angle must be between 0 and 180 degrees.")
        self._threshold_angle_rad = math.radians(new_angle_degrees)

    def forward(self, degradation_aware_prompts):
        """
        Calculates the DDL.

        Args:
            degradation_aware_prompts (torch.Tensor): A tensor of shape
                [num_degradations, prompt_dim] containing the prompt vectors.

        Returns:
            torch.Tensor: The scalar DDL value.
        """
        num_degradations, _ = degradation_aware_prompts.shape # prompt_dim not used directly here
        if num_degradations < 2:
            # Loss is undefined or zero if less than 2 prompts
            return torch.tensor(0.0, device=degradation_aware_prompts.device)

        total_loss = 0.0
        num_pairs = 0
        for i in range(num_degradations):
            for j in range(i + 1, num_degradations):
                prompt_i = degradation_aware_prompts[i]
                prompt_j = degradation_aware_prompts[j]

                # Calculate cosine similarity and angle between the pair of prompts
                cos_sim = F.cosine_similarity(
                    prompt_i.unsqueeze(0), prompt_j.unsqueeze(0), dim=1, eps=self.epsilon
                )
                # Clamp cosine similarity to avoid NaN in acos due to precision issues
                angle_ij_rad = torch.acos(
                    torch.clamp(cos_sim, -1.0 + self.epsilon, 1.0 - self.epsilon)
                )
                
                # Penalize if the angle is less than the threshold
                pair_loss = torch.clamp(self._threshold_angle_rad - angle_ij_rad, min=0.0)
                total_loss += pair_loss
                num_pairs += 1
        
        if num_pairs == 0: # Should not happen if num_degradations >= 2
            return torch.tensor(0.0, device=degradation_aware_prompts.device)

        # Normalize the loss by the number of pairs
        # The paper's normalization factor: 2 / (N * (N-1))
        normalization_factor = 2.0 / (num_degradations * (num_degradations - 1)) \
                               if num_degradations > 1 else 1.0
        return normalization_factor * total_loss.squeeze()


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss.

    Computes L1 loss between feature maps extracted from specified layers
    of a pre-trained VGG19 network.
    """
    def __init__(self, feature_layers=(2, 7, 16, 25, 34), use_input_norm=True, requires_grad=False):
        """
        Args:
            feature_layers (list of int): Indices of VGG19 layers to extract features from.
                                          Default corresponds to relu1_1, relu2_1, relu3_1, relu4_1, relu5_1.
            use_input_norm (bool): If True, normalizes input images using ImageNet mean/std.
            requires_grad (bool): If False, VGG parameters are frozen.
        """
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_layers = feature_layers
        
        if use_input_norm:
            # Standard ImageNet normalization parameters
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.use_input_norm = use_input_norm
        
        if not requires_grad:
            for param in self.vgg.parameters():
                param.requires_grad = False
        
        self.loss_fn = nn.L1Loss()

    def _normalize_input(self, x):
        """Normalizes the input tensor if use_input_norm is True."""
        if not self.use_input_norm:
            return x
        # Ensure mean and std are on the same device as input
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return (x - self.mean) / self.std

    def forward(self, input_img, target_img):
        """
        Calculates the perceptual loss.

        Args:
            input_img (torch.Tensor): The input image tensor [B, C, H, W].
            target_img (torch.Tensor): The target image tensor [B, C, H, W].

        Returns:
            torch.Tensor: The scalar perceptual loss value.
        """
        input_norm = self._normalize_input(input_img)
        target_norm = self._normalize_input(target_img)
        
        x_in, x_tgt = input_norm, target_norm
        current_loss = 0.0
        for i, layer in enumerate(self.vgg):
            x_in = layer(x_in)
            x_tgt = layer(x_tgt)
            if i in self.feature_layers:
                current_loss += self.loss_fn(x_in, x_tgt)
        
        return current_loss / len(self.feature_layers) if self.feature_layers else torch.tensor(0.0)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (a robust L1 variant).
    Loss = mean(sqrt( (x-y)^2 + eps^2 ))
    """
    def __init__(self, eps=1e-3):
        """
        Args:
            eps (float): A small constant for numerical stability and to make the
                         loss differentiable at zero. PromptIR paper uses 1e-3.
        """
        super().__init__()
        self.eps_squared = eps * eps # Pre-square epsilon for efficiency

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Predicted tensor.
            y (torch.Tensor): Target tensor.
        Returns:
            torch.Tensor: Scalar Charbonnier loss.
        """
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps_squared)
        return torch.mean(loss)


# --- SSIM Loss Components ---
def _gaussian(window_size, sigma):
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor(
        [math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    """Creates a 2D Gaussian window for SSIM calculation."""
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim_core(img1, img2, window, window_size, channel, data_range=1.0,
               size_average=True, C1_k=0.01, C2_k=0.03):
    """Core SSIM calculation function."""
    C1 = (C1_k * data_range)**2
    C2 = (C2_k * data_range)**2

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Ensure sigma values are non-negative before any division
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        # Return mean SSIM per image in batch
        return ssim_map.mean([1, 2, 3]) # Mean over C, H, W


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) Loss.
    Loss = 1 - SSIM_score.
    """
    def __init__(self, window_size=11, size_average=True, data_range=1.0, channel=3):
        """
        Args:
            window_size (int): Size of the Gaussian window.
            size_average (bool): If True, returns the mean SSIM score over the batch.
                                 Otherwise, returns SSIM per image in the batch.
            data_range (float): The dynamic range of the input image data (e.g., 1.0 for [0,1], 255 for [0,255]).
            channel (int): Number of channels in the input images.
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.data_range = data_range
        self.channel = channel
        self.window = _create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        Args:
            img1 (torch.Tensor): First image tensor [B, C, H, W].
            img2 (torch.Tensor): Second image tensor [B, C, H, W].
        Returns:
            torch.Tensor: Scalar SSIM loss.
        """
        # Ensure window is on the same device as images
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        ssim_val = _ssim_core(
            img1, img2, self.window, self.window_size, self.channel,
            self.data_range, self.size_average
        )
        return 1.0 - ssim_val  # Loss is 1 - SSIM


if __name__ == '__main__':
    # --- Test DirectionalDecoupledLoss ---
    print("\n--- Testing DirectionalDecoupledLoss ---")
    num_degradations_test = 2
    prompt_dimension_test = 64
    threshold_deg_test = 90.0
    loss_fn_ddl_test = DirectionalDecoupledLoss(threshold_angle_degrees=threshold_deg_test)
    
    # Test with orthogonal prompts (angle = 90 deg, loss should be 0)
    prompts_ortho = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32) * 5
    if prompt_dimension_test > 2:
        prompts_ortho = F.pad(prompts_ortho, (0, prompt_dimension_test - 2))
    print(f"DDL (orthogonal prompts, threshold {threshold_deg_test}°): "
          f"{loss_fn_ddl_test(prompts_ortho).item():.4f}")

    # Test with aligned prompts (angle = 0 deg, loss should be threshold_angle_rad)
    prompts_aligned = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32) * 5
    if prompt_dimension_test > 2:
        prompts_aligned = F.pad(prompts_aligned, (0, prompt_dimension_test - 2))
    print(f"DDL (aligned prompts, threshold {threshold_deg_test}°): "
          f"{loss_fn_ddl_test(prompts_aligned).item():.4f} "
          f"(Expected approx: {math.radians(threshold_deg_test):.4f})")

    # --- Test VGGPerceptualLoss ---
    print("\n--- Testing VGGPerceptualLoss ---")
    device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual_loss_fn_test = VGGPerceptualLoss().to(device_test)
    img1_test = torch.rand(2, 3, 64, 64, device=device_test)
    img2_test = img1_test.clone() * 0.5  # Different image
    print(f"Perceptual loss (different images): {perceptual_loss_fn_test(img1_test, img2_test).item():.4f}")
    print(f"Perceptual loss (same images): {perceptual_loss_fn_test(img1_test, img1_test.clone()).item():.4f}")

    # --- Test CharbonnierLoss ---
    print("\n--- Testing CharbonnierLoss ---")
    char_loss_fn_test = CharbonnierLoss(eps=1e-3).to(device_test)
    print(f"Charbonnier loss (different images): {char_loss_fn_test(img1_test, img2_test).item():.4f}")
    print(f"Charbonnier loss (same images): {char_loss_fn_test(img1_test, img1_test.clone()).item():.4f}")

    # --- Test SSIMLoss ---
    print("\n--- Testing SSIMLoss ---")
    ssim_loss_fn_test = SSIMLoss(window_size=11, data_range=1.0).to(device_test)
    print(f"SSIM loss (different images, 1-SSIM): {ssim_loss_fn_test(img1_test, img2_test).item():.4f}")
    print(f"SSIM loss (same images, 1-SSIM): {ssim_loss_fn_test(img1_test, img1_test.clone()).item():.4f}")
