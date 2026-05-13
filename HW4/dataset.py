"""
Provides the ImageRestorationDataset class for loading and preprocessing
image pairs (degraded and clean) for image restoration tasks.
Includes functionality for random cropping, data augmentation, and handling
different degradation types (e.g., rain, snow).
"""
import os
import glob
import random
import time  # For __main__ test

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


# --- Augmentation functions (inspired by a common image utility style) ---
def _augment_patch_numpy(image_np, mode):
    """
    Applies a specific geometric augmentation to a NumPy image array (H, W, C).

    Args:
        image_np (np.ndarray): Input image as a NumPy array (H, W, C).
        mode (int): Augmentation mode (0-7).
            0: original
            1: flip up and down
            2: rotate counter-clockwise 90 degrees
            3: rotate 90 degrees and flip up and down
            4: rotate 180 degrees
            5: rotate 180 degrees and flip
            6: rotate 270 degrees
            7: rotate 270 degrees and flip

    Returns:
        np.ndarray: Augmented image as a NumPy array.
    """
    if mode == 0:  # original
        return image_np
    elif mode == 1:  # flip up and down
        return np.flipud(image_np)
    elif mode == 2:  # rotate counter-clockwise 90 degrees
        return np.rot90(image_np, k=1, axes=(0, 1))  # H,W axes
    elif mode == 3:  # rotate 90 degrees and flip up and down
        out = np.rot90(image_np, k=1, axes=(0, 1))
        return np.flipud(out)
    elif mode == 4:  # rotate 180 degrees
        return np.rot90(image_np, k=2, axes=(0, 1))
    elif mode == 5:  # rotate 180 degrees and flip
        out = np.rot90(image_np, k=2, axes=(0, 1))
        return np.flipud(out)
    elif mode == 6:  # rotate 270 degrees
        return np.rot90(image_np, k=3, axes=(0, 1))
    elif mode == 7:  # rotate 270 degrees and flip
        out = np.rot90(image_np, k=3, axes=(0, 1))
        return np.flipud(out)
    return image_np


def _random_augment_paired_patches_numpy(patch1_np, patch2_np):
    """
    Applies the same randomly chosen geometric augmentation to two NumPy patches.

    Args:
        patch1_np (np.ndarray): First NumPy patch (H, W, C).
        patch2_np (np.ndarray): Second NumPy patch (H, W, C).

    Returns:
        tuple: A tuple containing the two augmented NumPy patches.
    """
    augmentation_mode = random.randint(0, 7)  # 0: original, 1-7: augmentations
    out1 = _augment_patch_numpy(patch1_np, augmentation_mode).copy() # Use .copy() to ensure new array
    out2 = _augment_patch_numpy(patch2_np, augmentation_mode).copy() # Use .copy() to ensure new array
    return out1, out2


class ImageRestorationDataset(Dataset):
    """
    Dataset for image restoration tasks. Loads pairs of degraded and clean images.
    Handles different degradation types (rain, snow) and applies transformations.
    In training mode, it performs random cropping and data augmentation.
    In validation/test mode, it resizes images to a specified size.
    """
    def __init__(self, degraded_base_dir, clean_base_dir, patch_size=128,
                 is_train=True, oversample_factor=1):
        """
        Args:
            degraded_base_dir (str): Base directory for degraded images.
            clean_base_dir (str): Base directory for clean images.
            patch_size (int): Size of the random patches to crop (if is_train=True).
                              If is_train=False, images are resized to this size.
            is_train (bool): If True, applies random cropping and augmentations.
                             If False, resizes the whole image.
            oversample_factor (int): Factor to multiply the dataset size by for training
                                     (repeats and shuffles the base file list).
        """
        self.degraded_base_dir = degraded_base_dir
        self.clean_base_dir = clean_base_dir
        self.patch_size = patch_size
        self.is_train = is_train
        self.oversample_factor = oversample_factor

        # Gather all degraded image paths and assign a type label
        degraded_rain_files = sorted(glob.glob(os.path.join(degraded_base_dir, "rain-*.png")))
        degraded_snow_files = sorted(glob.glob(os.path.join(degraded_base_dir, "snow-*.png")))
        
        self.base_file_list = []
        for f_path in degraded_rain_files:
            self.base_file_list.append({"path": f_path, "type": 0})  # 0 for rain
        for f_path in degraded_snow_files:
            self.base_file_list.append({"path": f_path, "type": 1})  # 1 for snow

        # Apply oversampling and shuffling for training
        if self.is_train:
            self.file_list = self.base_file_list * self.oversample_factor
            if self.oversample_factor > 1 and len(self.base_file_list) > 0:
                random.shuffle(self.file_list) # Shuffle after oversampling
        else:  # For validation/testing, use the original list without oversampling
            self.file_list = self.base_file_list
            
        # Transformation to convert NumPy array (HWC, uint8) to PyTorch tensor (CHW, float [0,1])
        self.to_tensor = T.ToTensor()
        
        # Specific transform for validation/test mode (resize whole image)
        if not self.is_train:
            self.resize_transform = T.Compose([
                T.ToPILImage(),  # Ensure input is PIL Image for Resize
                T.Resize((patch_size, patch_size)),
                T.ToTensor()
            ])

        # Create a mapping from degraded image paths to their corresponding clean image paths
        self.clean_file_map = self._create_clean_file_map()

    def _create_clean_file_map(self):
        """
        Builds a dictionary mapping degraded image paths to their clean counterparts.
        This map is built once during initialization for efficiency.
        """
        clean_map = {}
        # Iterate over the unique base file list to avoid redundant checks
        for item in self.base_file_list:
            deg_file_path = item["path"]
            basename = os.path.basename(deg_file_path)
            
            clean_basename = ""
            if basename.startswith("rain-"):
                clean_basename = basename.replace("rain-", "rain_clean-")
            elif basename.startswith("snow-"):
                clean_basename = basename.replace("snow-", "snow_clean-")
            else:
                print(f"Warning: Unknown degraded file prefix for {basename} in _create_clean_file_map.")
                continue  # Skip if prefix is unknown
            
            clean_file_path = os.path.join(self.clean_base_dir, clean_basename)
            if os.path.exists(clean_file_path):
                clean_map[deg_file_path] = clean_file_path
            else:
                # Log a warning if a clean counterpart is not found
                print(f"Warning: Clean file not found for {deg_file_path} "
                      f"(expected at {clean_file_path}) in _create_clean_file_map.")
        return clean_map

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (degraded image, clean image, degradation type label).

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (degraded_tensor, clean_tensor, degradation_type_label)
        """
        item_info = self.file_list[idx]
        degraded_img_path = item_info["path"]
        degradation_type_label = torch.tensor(item_info["type"], dtype=torch.long)
        
        clean_img_path = self.clean_file_map.get(degraded_img_path)
        if clean_img_path is None:
            # This should ideally not happen if _create_clean_file_map handles missing files
            raise FileNotFoundError(
                f"Clean image not found in map for degraded image: {degraded_img_path}. "
                "Check dataset integrity and paths."
            )

        try:
            degraded_pil = Image.open(degraded_img_path).convert("RGB")
            clean_pil = Image.open(clean_img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading PIL image: {degraded_img_path} or {clean_img_path}. Error: {e}")
            # Propagate error or return a placeholder if robust error handling is needed downstream
            raise e

        if self.is_train:
            # Convert PIL images to NumPy arrays for cropping and augmentation
            degraded_np = np.array(degraded_pil)  # HWC, uint8
            clean_np = np.array(clean_pil)        # HWC, uint8

            # Random crop
            h_img, w_img, _ = degraded_np.shape
            if h_img < self.patch_size or w_img < self.patch_size:
                # If image is smaller than patch_size, resize it first.
                # This scenario should be rare if input images are consistently sized.
                degraded_pil_resized = degraded_pil.resize((self.patch_size, self.patch_size), Image.BICUBIC)
                clean_pil_resized = clean_pil.resize((self.patch_size, self.patch_size), Image.BICUBIC)
                degraded_np = np.array(degraded_pil_resized)
                clean_np = np.array(clean_pil_resized)
                h_img, w_img, _ = degraded_np.shape  # Update dimensions

            rand_h_start = random.randint(0, h_img - self.patch_size)
            rand_w_start = random.randint(0, w_img - self.patch_size)
            
            degraded_patch_np = degraded_np[
                rand_h_start : rand_h_start + self.patch_size,
                rand_w_start : rand_w_start + self.patch_size, :
            ]
            clean_patch_np = clean_np[
                rand_h_start : rand_h_start + self.patch_size,
                rand_w_start : rand_w_start + self.patch_size, :
            ]

            # Apply random geometric augmentations (flips, rotations)
            degraded_patch_aug_np, clean_patch_aug_np = _random_augment_paired_patches_numpy(
                degraded_patch_np, clean_patch_np
            )
                
            # Convert augmented NumPy patches to PyTorch tensors
            degraded_tensor = self.to_tensor(degraded_patch_aug_np)
            clean_tensor = self.to_tensor(clean_patch_aug_np)
        else:
            # For validation/test: resize the whole image to patch_size x patch_size
            degraded_tensor = self.resize_transform(np.array(degraded_pil))
            clean_tensor = self.resize_transform(np.array(clean_pil))
            
        return degraded_tensor, clean_tensor, degradation_type_label


if __name__ == '__main__':
    # Example Usage and Basic Test:
    print("\n--- Testing ImageRestorationDataset ---")
    
    # Define relative paths from the project root
    # Assumes script is run from HW#04/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = current_script_dir
    
    train_degraded_dir_test = os.path.join(project_root_dir, "Data/train/degraded")
    train_clean_dir_test = os.path.join(project_root_dir, "Data/train/clean")

    print(f"Checking existence of: {train_degraded_dir_test}")
    print(f"Exists: {os.path.exists(train_degraded_dir_test)}")
    print(f"Checking existence of: {train_clean_dir_test}")
    print(f"Exists: {os.path.exists(train_clean_dir_test)}")

    if os.path.exists(train_degraded_dir_test):
        print(f"Contents of {train_degraded_dir_test} (first 5): {os.listdir(train_degraded_dir_test)[:5]}")
    if os.path.exists(train_clean_dir_test):
         print(f"Contents of {train_clean_dir_test} (first 5): {os.listdir(train_clean_dir_test)[:5]}")

    try:
        print("\n--- Initializing Training Dataset (patch_size=128) ---")
        train_dataset_test = ImageRestorationDataset(
            degraded_base_dir=train_degraded_dir_test,
            clean_base_dir=train_clean_dir_test,
            patch_size=128,
            is_train=True,
            oversample_factor=2 # Example oversampling
        )
        print(f"Training dataset length (with oversampling): {len(train_dataset_test)}")

        if len(train_dataset_test) > 0:
            start_time = time.time()
            degraded_sample_train, clean_sample_train, label_sample_train = train_dataset_test[0]
            end_time = time.time()
            print(f"Time to fetch one training sample: {end_time - start_time:.4f}s")
            print(f"Degraded training sample shape: {degraded_sample_train.shape}")
            print(f"Clean training sample shape: {clean_sample_train.shape}")
            print(f"Training label sample: {label_sample_train}")
            assert degraded_sample_train.shape == (3, 128, 128)
            assert clean_sample_train.shape == (3, 128, 128)
        else:
            print("Training dataset is empty. Check paths and file structure.")

        print("\n--- Initializing Validation Dataset (resize_to_size=256) ---")
        val_dataset_test = ImageRestorationDataset(
            degraded_base_dir=train_degraded_dir_test, # Using train data for test purposes here
            clean_base_dir=train_clean_dir_test,
            patch_size=256, # This will be the resize target
            is_train=False
        )
        print(f"Validation dataset length: {len(val_dataset_test)}")
        if len(val_dataset_test) > 0:
            start_time = time.time()
            degraded_sample_val, clean_sample_val, label_sample_val = val_dataset_test[0]
            end_time = time.time()
            print(f"Time to fetch one validation sample: {end_time - start_time:.4f}s")
            print(f"Degraded validation sample shape: {degraded_sample_val.shape}")
            print(f"Clean validation sample shape: {clean_sample_val.shape}")
            print(f"Validation label sample: {label_sample_val}")
            assert degraded_sample_val.shape == (3, 256, 256)
            assert clean_sample_val.shape == (3, 256, 256)
        else:
            print("Validation dataset is empty.")
            
        print("\nDataset class basic tests seem to pass.")

    except Exception as e:
        print(f"Error during dataset test: {e}")
        import traceback
        traceback.print_exc()
