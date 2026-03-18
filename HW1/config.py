"""
Configuration for the 100-class image classification project.
"""

# Normalization statistics
MEAN = [0.4575, 0.4705, 0.3730]
STD = [0.1975, 0.1955, 0.2001]

# Input image size
IMAGE_SIZE = 512

# Training settings
BATCH_SIZE = 12
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_PROB = 0.2
PATIENCE = 15

# Augmentation settings
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.2
COLOR_JITTER = 0.2
ROTATION_DEGREES = 15

# Number of classes
NUM_CLASSES = 100

# Default paths
DEFAULT_TRAIN_DIR = "./data/train"
DEFAULT_VAL_DIR = "./data/val"
DEFAULT_TEST_DIR = "./data/test"
DEFAULT_SAVE_DIR = "./results"

# TTA settings
TTA_NUM_AUGS = 5