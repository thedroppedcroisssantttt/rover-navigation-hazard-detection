# Hazard Rock Detection for Rovers Using Synthetic Training Data

## Overview

Autonomous rovers must recognize large rocks that could block their path or damage their wheels. Training a detector for this task requires labeled images, and real labeled images of the lunar surface are scarce. This project trains a YOLOv8 detector on rendered lunar scenes only and evaluates it on real lunar photographs. I measure how much accuracy is lost between rendered and real images, then test whether photometric augmentations of the training renders reduce that gap.

## Dataset

I use the Artificial Lunar Landscape Dataset (Pessia, 2019), available on Kaggle under a CC BY-NC-SA 4.0 license. It contains 9,766 rendered scenes and 36 real photographs. The renders are split into training and validation sets. The real photographs are held out as a test set and are never used for training. The dataset is not included in this repository. Download it from Kaggle and unzip it into `data/raw/lunar/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the following from the project root.

1. `python scripts/convertLabels.py` converts the bounding boxes to YOLO format.
2. `python scripts/splitData.py` splits the renders into training and validation sets and writes `data.yaml`.
3. `python scripts/checkLabels.py` draws the converted boxes on sample images for a visual check.
4. `python scripts/augment.py --aug <name>` builds an augmented training set, where `<name>` is `contrast`, `blur`, `noise`, `lighting`, or `all`.
5. `notebooks/trainColab.ipynb` trains and evaluates one configuration per run in Google Colab.
6. `python scripts/visualize.py` produces the figures and results table.

## Repository Structure

```
scripts/      data preparation, augmentation, and figures
notebooks/    training notebook
results/      experiment log and figures
docs/         project page
space/        demo application
```

The `data/` and `runs/` folders are excluded from version control.

## References

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

Pessia, R. (2019). *Artificial lunar landscape dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset

Portions of this work were developed with the assistance of generative AI tools. I reviewed all outputs and take full responsibility for the content.
