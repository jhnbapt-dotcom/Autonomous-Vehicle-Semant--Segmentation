import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CamVidSegmentation
from model import create_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3 on the CamVid dataset."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to the CamVid folder containing train, val, and test folders.",
    )
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = CamVidSegmentation(args.data_root, split="train")
    val_dataset = CamVidSegmentation(args.data_root, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(
        num_classes=args.num_classes,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{args.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                args.output_dir,
                "best_deeplabv3_camvid.pth",
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
