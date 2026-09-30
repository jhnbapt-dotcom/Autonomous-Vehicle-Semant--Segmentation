# Autonomous Vehicle Semantic Segmentation

A PyTorch project for semantic segmentation of autonomous-driving scenes using the CamVid dataset and DeepLabV3 with a ResNet-50 backbone.

## Goal

Semantic segmentation predicts a class for every pixel in a road-scene image. It helps autonomous vehicles distinguish road surfaces, cars, pedestrians, buildings, sky, vegetation, sidewalks, and other scene elements.

## Model

- Model: DeepLabV3
- Backbone: ResNet-50
- Framework: PyTorch and torchvision
- Default number of classes: 11
- Image size: 256 × 256
- Loss: Cross-entropy loss with ignore index `255`

## Dataset

This repository does **not** include CamVid files.

The expected prepared structure is:

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

See [`data/README.md`](data/README.md) for dataset notes.

## Installation

```bash
git clone [https://github.com/jhnbapt-dotcom/Autonomous-Vehicle-Semant--Segmentation.git](https://github.com/jhnbapt-dotcom/Autonomous-Vehicle-Semant--Segmentation.git)
cd Autonomous-Vehicle-Semant--Segmentation
pip install -r requirements.txt
```

## Training

Run the following command after preparing your CamVid folders:

```bash
python src/train.py \
  --data_root /path/to/camvid \
  --num_epochs 10 \
  --batch_size 4
```

The best model checkpoint is saved as:

```text
outputs/best_deeplabv3_camvid.pth
```

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── src/
    ├── __init__.py
    ├── dataset.py
    ├── model.py
    └── train.py
```

## Next improvements

- Verify the CamVid label-file naming convention.
- Add mean Intersection over Union (mIoU) evaluation.
- Add prediction visualization.
- Add a notebook for Google Colab.
- Add model inference for a single driving-scene image.
