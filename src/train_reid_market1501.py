"""
train_reid_market1501.py

Portfolio-style training script for person re-identification (ReID)
on the Market-1501 dataset using a Siamese-style network and triplet loss.

This is a simplified, clean version of the original Colab notebook code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


@dataclass
class ReIDConfig:
    data_root: Path = Path("data/market1501")
    train_dir_name: str = "bounding_box_train"
    emb_dim: int = 256
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    margin: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: Path = Path("checkpoints/reid_market1501.pth")


def set_seed(seed: int) -> None:
    import random
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Market1501TripletDataset(Dataset):
    """
    Triplet sampling dataset for Market-1501.

    Expects a folder structure like:
        data_root/train_dir_name/
            0001_c1s1_000151_01.jpg
            0001_c1s1_000176_01.jpg
            0002_c1s1_000251_01.jpg
            ...

    Where the first four characters of the filename represent the person ID (PID).
    """

    def __init__(self, root: Path, transform=None):
        self.root = root
        self.transform = transform

        self.img_paths: List[Path] = sorted([p for p in root.glob("*.jpg")])
        if not self.img_paths:
            raise FileNotFoundError(f"No .jpg files found in {root}")

        # Extract person IDs from filenames (first 4 characters by convention)
        self.pids: List[int] = []
        for p in self.img_paths:
            pid_str = p.name.split("_")[0]
            # Market-1501 uses -1 for "junk" images; skip them
            pid = int(pid_str)
            if pid == -1:
                continue
            self.pids.append(pid)

        # Reconstruct img_paths to match filtered PIDs (simple implementation)
        self.img_paths = [p for p in self.img_paths if int(p.name.split("_")[0]) != -1]

        # Map PID -> list of indices
        self.pid2indices = {}
        for idx, p in enumerate(self.img_paths):
            pid = int(p.name.split("_")[0])
            self.pid2indices.setdefault(pid, []).append(idx)

        self.unique_pids = sorted(self.pid2indices.keys())
        if len(self.unique_pids) < 2:
            raise ValueError("Need at least 2 unique person IDs for triplet sampling.")

    def __len__(self) -> int:
        return len(self.img_paths)

    def _sample_positive(self, pid: int, anchor_idx: int) -> int:
        indices = self.pid2indices[pid]
        if len(indices) == 1:
            return anchor_idx
        pos_idx = anchor_idx
        while pos_idx == anchor_idx:
            pos_idx = np.random.choice(indices)
        return pos_idx

    def _sample_negative(self, pid: int) -> int:
        neg_pid = pid
        while neg_pid == pid:
            neg_pid = np.random.choice(self.unique_pids)
        neg_idx = np.random.choice(self.pid2indices[neg_pid])
        return neg_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path = self.img_paths[idx]
        anchor_pid = int(anchor_path.name.split("_")[0])

        pos_idx = self._sample_positive(anchor_pid, idx)
        neg_idx = self._sample_negative(anchor_pid)

        pos_path = self.img_paths[pos_idx]
        neg_path = self.img_paths[neg_idx]

        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img = Image.open(pos_path).convert("RGB")
        neg_img = Image.open(neg_path).convert("RGB")

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img


class ReIDEmbeddingNet(nn.Module):
    """
    ResNet-50 backbone + projection head that outputs L2-normalized embeddings.
    """

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # remove original classification head
        self.backbone = backbone
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        emb = self.embedding_head(feats)
        # L2 normalize
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


def get_transforms_train() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((256, 128), padding=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_dataloader(cfg: ReIDConfig) -> DataLoader:
    train_root = cfg.data_root / cfg.train_dir_name
    ds = Market1501TripletDataset(train_root, transform=get_transforms_train())
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    print(f"Loaded Market-1501 train set with {len(ds)} images and {len(ds.unique_pids)} unique PIDs.")
    return dl


def train_reid(cfg: ReIDConfig) -> nn.Module:
    set_seed(cfg.seed)

    device = cfg.device
    print(f"Using device: {device}")

    dataloader = build_dataloader(cfg)
    model = ReIDEmbeddingNet(emb_dim=cfg.emb_dim).to(device)

    criterion = nn.TripletMarginLoss(margin=cfg.margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for anchors, positives, negatives in tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.num_epochs}"):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            anc_emb = model(anchors)
            pos_emb = model(positives)
            neg_emb = model(negatives)

            loss = criterion(anc_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(1, num_batches)
        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}")

        # Save latest weights (could also implement best-loss tracking)
        torch.save(model.state_dict(), cfg.save_path)
        print(f"Saved model checkpoint to {cfg.save_path}")

    return model


if __name__ == "__main__":
    cfg = ReIDConfig()
    train_reid(cfg)
