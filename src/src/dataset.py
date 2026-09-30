import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class CamVidSegmentation(Dataset):
    """
    CamVid semantic-segmentation dataset.

    Expected folder structure:
    camvid/
      train/images, train/labels
      val/images, val/labels
      test/images, test/labels
    """

    def __init__(self, root, split="train", image_size=(256, 256)):
        self.image_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels")
        self.image_size = image_size

        self.image_files = sorted(
            file_name
            for file_name in os.listdir(self.image_dir)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name)

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

        label = torch.tensor(
            list(label.getdata()),
            dtype=torch.long,
        ).view(self.image_size[0], self.image_size[1])

        return image, label
