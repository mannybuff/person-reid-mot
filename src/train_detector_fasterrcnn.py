"""
train_detector_fasterrcnn.py

Portfolio-style training script for a pedestrian detector using
Faster R-CNN (ResNet-50 FPN backbone) on a MOT-style dataset.

This script assumes you have MOT-style frames and annotations and
focuses on a clean, readable training loop rather than full MOT logic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.ops import box_convert
from PIL import Image
from tqdm import tqdm


@dataclass
class DetConfig:
    data_root: Path = Path("data/mot")
    train_images_dir: str = "train/images"
    train_ann_path: str = "train/annotations.txt"
    val_images_dir: str = "val/images"
    val_ann_path: str = "val/annotations.txt"
    num_classes: int = 2  # background + person
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_path: Path = Path("checkpoints/fasterrcnn_pedestrian_mot.pth")


def set_seed(seed: int) -> None:
    import random
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class MOTDetectionDataset(Dataset):
    """
    Minimal MOT-style detection dataset.

    Expects:
        - images in an images/ folder
        - annotations in a simple txt format:

            frame_id, x, y, w, h

        where coordinates are in pixels and correspond to 'person' boxes.
        (This is intentionally simplified for a portfolio example.)

    You can adapt this loader to your exact MOT annotation format as needed.
    """

    def __init__(self, images_dir: Path, ann_path: Path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        # Read annotations: frame -> list of boxes
        self.frame_to_boxes: Dict[str, List[List[float]]] = {}
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Example row: frame_000001.jpg, x, y, w, h
                parts = line.split(",")
                frame_name = parts[0].strip()
                x, y, w, h = map(float, parts[1:5])
                self.frame_to_boxes.setdefault(frame_name, []).append([x, y, w, h])

        self.frame_names = sorted(self.frame_to_boxes.keys())

    def __len__(self) -> int:
        return len(self.frame_names)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        frame_name = self.frame_names[idx]
        img_path = self.images_dir / frame_name
        img = Image.open(img_path).convert("RGB")

        boxes_wh = np.array(self.frame_to_boxes[frame_name], dtype=np.float32)
        if boxes_wh.size == 0:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        else:
            # Convert [x, y, w, h] -> [x1, y1, x2, y2]
            boxes_xyxy = box_convert(
                torch.from_numpy(boxes_wh),
                in_fmt="xywh",
                out_fmt="xyxy",
            ).numpy()

        labels = np.ones((boxes_xyxy.shape[0],), dtype=np.int64)  # all 'person'

        if self.transform is not None:
            img = self.transform(img)

        target = {
            "boxes": torch.as_tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return img, target


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
    ])


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def build_dataloaders(cfg: DetConfig) -> Tuple[DataLoader, DataLoader]:
    train_images = cfg.data_root / cfg.train_images_dir
    train_ann = cfg.data_root / cfg.train_ann_path
    val_images = cfg.data_root / cfg.val_images_dir
    val_ann = cfg.data_root / cfg.val_ann_path

    transform = get_transforms()

    train_ds = MOTDetectionDataset(train_images, train_ann, transform=transform)
    val_ds = MOTDetectionDataset(val_images, val_ann, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return train_loader, val_loader


def build_model(cfg: DetConfig) -> nn.Module:
    """
    Build a Faster R-CNN model for single-class detection (person).
    """
    model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    # Replace the head for our number of classes (background + 1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        cfg.num_classes,
    )
    return model


def train_detector(cfg: DetConfig) -> nn.Module:
    set_seed(cfg.seed)
    device = cfg.device
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    # Only parameters of the head are fine-tuned by default
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} [train]"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}")

        # Simple validation loop (no gradients)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} [val]"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch}: val_loss={avg_val_loss:.4f}")

        # Save last checkpoint each epoch (could track best)
        torch.save(model.state_dict(), cfg.save_path)
        print(f"Saved model checkpoint to {cfg.save_path}")

    return model


if __name__ == "__main__":
    cfg = DetConfig()
    train_detector(cfg)
