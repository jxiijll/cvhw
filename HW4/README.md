# NYCU Visual Recognition HW4: Image Restoration

Student ID: 414551008  
Name: 鄧浩培

This repository contains the source code for HW4 image restoration on degraded weather images. The final submission is generated as a compressed NumPy file, `pred.npz`, where each key is the original test filename such as `0.png`, and each value is a restored RGB image stored in CHW format with shape `(3, 256, 256)` and dtype `uint8`.

## Introduction

The task is to restore clean images from degraded inputs containing rain or snow. This is challenging because the degradation can cover both low-frequency image content and high-frequency textures, while the restored output still needs to preserve color, structure, and local details.

The main method is based on a PromptIR-style U-Net architecture with Transformer blocks and dynamic prompt generation. The model learns prompt components for image restoration and can optionally use degradation-aware modules:

- Degradation classifier: an auxiliary rain/snow classifier from bottleneck features.
- Task prompt bank: separate shared, rain, and snow prompt banks routed by degradation probabilities.
- Frequency branch: a Laplacian-based high-frequency branch fused before refinement.

The default training objective combines pixel, structural, robust, and perceptual losses to balance PSNR and visual quality.

## Method

### Data Preprocessing

The dataset contains paired degraded and clean images. The training loader scans:

- `Data/train/degraded/rain-*.png`
- `Data/train/degraded/snow-*.png`

Each degraded image is paired with its corresponding clean target in `Data/train/clean/`:

- `rain-xxx.png` -> `rain_clean-xxx.png`
- `snow-xxx.png` -> `snow_clean-xxx.png`

During training, the dataset performs random 256x256 cropping and paired geometric augmentation. During validation and inference, images are resized to 256x256 and converted to tensors in the `[0, 1]` range.

### Model Architecture

The model is implemented in `promptir_model.py` and uses:

- Encoder-decoder U-Net structure
- Transformer blocks from `transformer_block.py`
- PixelShuffle upsampling
- Prompt generation blocks at decoder stages
- Prompt interaction blocks that fuse decoder features with learned prompts
- Optional rain/snow degradation classifier
- Optional task-specific prompt banks
- Optional high-frequency restoration branch

The default model configuration is:

- Input/output channels: 3
- Base channel dimension: 48
- Transformer blocks per level: `[3, 4, 4, 6]`
- Refinement blocks: 4
- Prompt components: 5
- Prompt dimensions: 256, 128, and 64 for deep, mid, and shallow decoder stages

### Training Strategy

The training script supports:

- AdamW optimizer with cosine annealing learning rate schedule
- 90/10 train/validation split with a fixed random seed
- L1, SSIM, Charbonnier, and VGG19 perceptual losses
- Gradient clipping
- Optional resume from `latest` or a checkpoint path
- Optional acceleration through Hugging Face `accelerate`
- Optional degradation-aware modules controlled by command-line flags

Validation reports L1 loss, PSNR, and degradation classification accuracy when the classifier is enabled.

## Environment Setup

Create an environment and install dependencies:

```bash
conda create -n cv python=3.10
conda activate cv
pip install -r requirements.txt
```

Main dependencies:

- PyTorch
- torchvision
- Pillow
- NumPy
- accelerate


## Dataset Structure

Place the dataset under `Data/`:

```text
Data/
  train/
    degraded/
      rain-0.png
      rain-1.png
      snow-0.png
      snow-1.png
      ...
    clean/
      rain_clean-0.png
      rain_clean-1.png
      snow_clean-0.png
      snow_clean-1.png
      ...
  test/
    degraded/
      0.png
      1.png
      ...
      99.png
```

Do not include the dataset or model checkpoints in the final submission zip.

## Usage

### Train Baseline PromptIR

```bash
python train.py --data-root Data
```

This saves checkpoints under `trained_models/`, including:

- `promptir_best.pth`
- `promptir_train_state_last.pth`
- `promptir_last_epoch{N}.pth`

### Resume Training

```bash
python train.py --data-root Data --resume latest
```

You can also resume from a specific checkpoint:

```bash
python train.py --data-root Data --resume trained_models/promptir_train_state_last.pth
```

### Train With Degradation-Aware Modules

```bash
python train.py --data-root Data \
  --use-degradation-classifier \
  --use-task-prompt-bank \
  --use-frequency-branch \
  --lambda-cls 0.05
```

These flags must match the checkpoint architecture when manually loading a model. The prediction and analysis scripts can also auto-enable modules detected in a checkpoint.

### Single-Model Inference

```bash
python predict.py \
  --data-root Data \
  --checkpoint trained_models/promptir_best.pth
```

This reads images from `Data/test/degraded/` and writes `pred.npz` in the project root. The script expects 100 test images named `0.png` through `99.png`; missing predictions are filled with zero placeholders so the output file keeps the required keys.

### Validation Analysis and Qualitative Figures

```bash
python analyze.py \
  --checkpoint trained_models/promptir_best.pth \
  --degraded-dir Data/train/degraded \
  --clean-dir Data/train/clean \
  --output-dir analysis \
  --num-visuals 5
```

This produces validation metrics and qualitative restoration panels:

```text
degraded | restored | clean | error x4
```

It also exports prompt-weight plots when prompt weights are available.

### Accelerate Training

If `accelerate` is configured, the training script can be launched with:

```bash
accelerate launch train.py --data-root Data
```

The script detects the accelerate environment and prepares the model, optimizer, and dataloaders accordingly.

## Results and Analysis

The validation workflow evaluates restoration quality with PSNR and SSIM-related metrics. The analysis script compares degraded input quality against restored output quality, reports rain/snow-wise performance, and saves qualitative images for best, worst, and sampled validation examples.

The optional degradation-aware modules are intended to test whether explicitly modeling rain and snow helps the prompt generator select more suitable restoration behavior. The frequency branch is intended to recover sharper details by injecting lightweight high-frequency cues before the refinement stage.

### Validation Curves

Training logs are printed to stdout. If external logging is needed, redirect the terminal output while training.

### Final Validation Comparison

Use `analysis/summary.json`, `analysis/metrics.csv`, and `analysis/degradation_wise_metrics.csv` to compare checkpoint performance.

### Qualitative Example

Qualitative panels are saved under:

```text
analysis/visuals/
analysis/prompt_plots/
```

### Public Leaderboard Snapshot

![Public leaderboard snapshot](images/benchmark.png)

## Code Structure

- `dataset.py`: paired rain/snow restoration dataset, clean-image matching, cropping, resizing, and augmentation
- `promptir_model.py`: PromptIR model, prompt generation, degradation classifier, task prompt banks, and frequency branch
- `transformer_block.py`: Transformer attention and feed-forward blocks used by PromptIR
- `train.py`: training, validation, checkpoint saving, resume logic, and optional accelerate support
- `predict.py`: test-set inference and `pred.npz` generation
- `analyze.py`: validation metrics, degradation-wise summaries, prompt-weight CSVs, and qualitative figures
- `losses.py`: L1-compatible restoration losses, SSIM loss, Charbonnier loss, VGG perceptual loss, and directional decoupling loss
- `metrics.py`: PSNR calculation helper
