[English](README.md) | [简体中文](README.zh.md)

# ResNet18 Mask Recognition

A PyTorch learning project for classifying images into **mask** and **no mask** categories. It directly implements residual blocks and a ResNet18-style network rather than loading a pretrained torchvision classifier.

The repository includes scripts, 1,030 dataset images, a saved `best_model.pth` checkpoint, sample images, and a [training figure](Figure_1.pdf). This is an image classification exercise; face localization, video processing, and a deployed application are outside its scope. No independently reproduced accuracy result is claimed here.

## Implementation

- Residual blocks with optional projection shortcuts and four residual stages.
- RGB inputs resized to 224 × 224 and normalized with fixed constants.
- Adam (`lr=0.001`), cross-entropy, batch size 32, and 50 epochs by default.
- Best checkpoint selection by validation accuracy and learning-curve plots.
- Test-set accuracy and a single-image prediction example.

## Environment

The original instructions suggest Python 3.9 or later. Dependencies are unpinned, so this is not a guaranteed compatibility range. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
# Activate .venv in your shell.
python -m pip install -r requirements.txt
```

Dependencies include PyTorch, torchvision, torchsummary, Pillow, NumPy, pandas, and Matplotlib. Select a PyTorch installation appropriate for your CPU/CUDA environment. Scripts select CUDA when available and otherwise use CPU.

Run commands from the repository root. Training/testing use Windows-style paths (`data\train`, `data\test`); adapt these on other operating systems. Training selects Matplotlib's `TkAgg` backend and opens a plot, requiring a desktop/Tk environment or a plotting adjustment for headless use.

## Data and workflow

The tracked dataset contains 500 images in `dataset/mask/` and 530 in `dataset/no_mask/`. `ImageFolder` assigns class indices by folder names: `mask`, then `no_mask`. Keep that ordering consistent with prediction labels.

### 1. Create the split

```bash
python data_partitioning.py
```

This copies 90% of each class into `data/train/` and 10% into `data/test/`, without a fixed random seed. Existing destination files are retained: rerunning over an earlier split can leave the same image in both sets. Use a fresh generated `data/` directory for each new split. Generated copies are excluded from Git.

### 2. Train, or use the existing checkpoint

```bash
python model_train.py
```

Training further divides `data/train/` into 80% training and 20% validation using unseeded `random_split`. It saves the parameters with the highest validation accuracy to `best_model.pth`, replacing the existing checkpoint, and displays learning curves. Preserve the bundled checkpoint separately if needed.

### 3. Evaluate and predict

```bash
python model_test.py
```

This loads `best_model.pth`, evaluates `data/test/`, and predicts the class of `images.jfif`. Prepare the test split even when using the bundled checkpoint. Change the image path in `model_test.py` for another RGB image. The original documentation describes the checkpoint as approximately 43 MB; its training split and results are not independently verified here.

To inspect the network summary:

```bash
python model.py
```

## Repository guide

```text
dataset/                Original mask/no_mask images
data_partitioning.py    Random 90:10 copy-based split
mean_std.py             Exploratory normalization-statistics script
model.py                Residual blocks and ResNet18-style classifier
model_train.py          Training, validation, checkpoint saving, plotting
model_test.py           Test accuracy and single-image prediction
best_model.pth          Existing model state dictionary
Figure_1.pdf            Existing training figure
images.jfif             Default prediction example
```

## Reproducibility and rights

`model_test.py` is a dataset evaluation script, not an automated unit test suite. No training, checkpoint loading, or evaluation was performed for this documentation update. Random splits mean results can vary.

Training and testing share fixed normalization constants. `mean_std.py` is exploratory: it divides per-channel totals by the total number of array elements and prints variance rather than standard deviation. Review the calculation before using it to replace normalization constants.

The repository does not identify the original dataset's source/license or contain a license file. Check applicable rights before redistributing images or weights; this repository provides no explicit open-source license grant.
