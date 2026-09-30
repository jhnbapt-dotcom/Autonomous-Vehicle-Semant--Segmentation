import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class CamVidSegmentation(Dataset):
    """
    CamVid semantic segmentation dataset.

    Expected prepared structure:

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
    """

    def __init__(self, root, split="train", image_size=(256, 256)):
        self.image_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels")
        self.image_size = image_size

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")

        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label folder not found: {self.label_dir}")

        self.image_files = sorted(
            file_name
            for file_name in os.listdir(self.image_dir)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not self.image_files:
            raise RuntimeError(f"No images found in: {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def _get_label_path(self, image_name):
        base_name, extension = os.path.splitext(image_name)

        possible_names = [
            image_name,
            f"{base_name}_L.png",
            f"{base_name}.png",
        ]

        for label_name in possible_names:
            label_path = os.path.join(self.label_dir, label_name)
            if os.path.exists(label_path):
                return label_path

        raise FileNotFoundError(
            f"No matching label found for {image_name} in {self.label_dir}"
        )

    def __getitem__(self, index):
        image_name = self.image_files[index]

        image_path = os.path.join(self.image_dir, image_name)
        label_path = self._get_label_path(image_name)

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
        )

        label = TF.resize(
            label,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )

        image = TF.to_tensor(image)

        # CamVid label masks must contain class IDs, not RGB colors.
        label = torch.tensor(
            list(label.getdata()),
            dtype=torch.long,
        ).view(self.image_size[0], self.image_size[1])

        return image, label
