# Autonomous Vehicle Semantic Segmentation

Semantic segmentation project for autonomous-driving scenes using the CamVid dataset, PyTorch, and DeepLabV3 with a ResNet-50 backbone.

## Objective

The model assigns a class label to every pixel in a road-scene image. Example classes include road, sky, building, vehicle, pedestrian, vegetation, and sidewalk.

## Dataset

This project uses CamVid image frames and semantic-label masks.

The dataset is not included in this repository because it is too large. Store it locally, in Google Drive, or download it separately.

Expected structure:

```text
camvid/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Model

- Architecture: DeepLabV3
- Backbone: ResNet-50
- Framework: PyTorch and torchvision
- Default classes: 11
- Input size: 256 × 256

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py --data_root /path/to/camvid --num_epochs 10
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
└── src/
    ├── __init__.py
    ├── dataset.py
    ├── model.py
    └── train.py
```
