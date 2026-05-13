"""
Provides utility functions for calculating image quality metrics.
Currently includes Peak Signal-to-Noise Ratio (PSNR).
"""
import torch
import math


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Images are expected to be PyTorch tensors. The function assumes that
    if max_val is 1.0, the input images are normalized to the [0, 1] range.
    If max_val is 255.0, images are expected in the [0, 255] range.

    Args:
        img1 (torch.Tensor): The first image tensor (e.g., restored image).
                             Shape can be [C, H, W] or [B, C, H, W].
        img2 (torch.Tensor): The second image tensor (e.g., clean ground truth).
                             Must have the same shape as img1.
        max_val (float): The maximum possible pixel value of the images.
                         Defaults to 1.0 for images normalized to [0,1].
                         Use 255.0 for images in the [0,255] range.

    Returns:
        float: The PSNR value in decibels (dB). Returns float('inf') if MSE is zero.
    """
    # Ensure tensors are float for MSE calculation
    img1 = img1.float()
    img2 = img2.float()

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        # PSNR is infinite if images are identical
        return float('inf')
    
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr


if __name__ == '__main__':
    # --- Example Usage and Basic Tests ---
    print("--- Testing calculate_psnr function ---")

    # Create two dummy images (e.g., BxCxHxW or CxHxW)
    # Test case 1: Perfect match (images normalized to [0, 1])
    img_a_norm = torch.rand(1, 3, 256, 256)
    img_b_norm = img_a_norm.clone()
    psnr_perfect = calculate_psnr(img_a_norm, img_b_norm, max_val=1.0)
    print(f"PSNR (perfect match, [0,1] range): {psnr_perfect:.2f} dB") # Expected: inf

    # Test case 2: Slight difference (images normalized to [0, 1])
    img_c_norm = img_a_norm.clone()
    img_c_norm += torch.randn_like(img_c_norm) * 0.01  # Add small noise
    img_c_norm = torch.clamp(img_c_norm, 0, 1)         # Clamp to [0,1] range
    psnr_slight_diff = calculate_psnr(img_a_norm, img_c_norm, max_val=1.0)
    print(f"PSNR (slight difference, [0,1] range): {psnr_slight_diff:.2f} dB")

    # Test case 3: Larger difference (images normalized to [0, 1])
    img_d_norm = torch.rand(1, 3, 256, 256)
    psnr_large_diff = calculate_psnr(img_a_norm, img_d_norm, max_val=1.0)
    print(f"PSNR (large difference, [0,1] range): {psnr_large_diff:.2f} dB")

    # Test case 4: Simulating uint8 images (0-255 range), but inputs to func are float
    img_e_uint_range = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.float32)
    img_f_uint_range = img_e_uint_range.clone()
    img_f_uint_range += torch.randint(-10, 11, img_f_uint_range.shape, dtype=torch.float32)
    img_f_uint_range = torch.clamp(img_f_uint_range, 0, 255)
    
    # Option A: Normalize images to [0,1] and use max_val=1.0
    psnr_uint_style_normalized = calculate_psnr(
        img_e_uint_range / 255.0, img_f_uint_range / 255.0, max_val=1.0
    )
    print(f"PSNR (uint8 style, inputs normalized to [0,1], max_val=1.0): {psnr_uint_style_normalized:.2f} dB")

    # Option B: Keep images in [0,255] float range and use max_val=255.0
    psnr_uint_direct_float = calculate_psnr(img_e_uint_range, img_f_uint_range, max_val=255.0)
    print(f"PSNR (uint8 style, inputs as float [0,255], max_val=255.0): {psnr_uint_direct_float:.2f} dB")
    
    # Note: For the competition, if output is uint8 (0-255), ensure consistency in how PSNR is calculated.
    # Using float tensors normalized to [0,1] with max_val=1.0 is a common practice.
    # If comparing with results that use uint8 images directly, ensure max_val=255.0 is used.
