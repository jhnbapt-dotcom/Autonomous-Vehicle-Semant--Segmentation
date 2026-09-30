import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


def create_model(num_classes=11, pretrained=True):
    """
    Creates DeepLabV3 with a ResNet-50 backbone.

    Args:
        num_classes: Number of output segmentation classes.
        pretrained: Use pretrained torchvision weights.

    Returns:
        A PyTorch DeepLabV3 segmentation model.
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None

    model = deeplabv3_resnet50(weights=weights)

    model.classifier[-1] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1,
    )

    return model
