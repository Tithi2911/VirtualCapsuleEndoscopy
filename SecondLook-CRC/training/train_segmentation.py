"""Train a polyp segmentation model and export it for SecondLook (ONNX + JSON sidecar).

Dataset layout (Kvasir-SEG, CVC-ClinicDB, PolypGen and VR-Caps exports all fit it):

    DATA/images/<name>.jpg|png
    DATA/masks/<name>.jpg|png      (white = polyp)

Typical sim-to-real recipe (as in Tasks/Disease Classification for VR-Caps):
    # 1. pre-train on VR-Caps synthetic frames
    python train_segmentation.py --data vrcaps_polyps --epochs 30 --out models/pretrain.pt
    # 2. fine-tune on real data and export
    python train_segmentation.py --data kvasir_seg --init models/pretrain.pt --epochs 40 \
        --out models/unet-kvasir.pt --export models/unet-kvasir.onnx

IMPORTANT: split train/validation by PATIENT (or by video), not by frame - frames
from the same patient on both sides inflate scores. Use --split-file to supply a
patient-level split. Check each dataset's licence before commercial use.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class PolypDataset(Dataset):
    def __init__(self, root: Path, names: list[str], size: int, augment: bool):
        self.root, self.names, self.size, self.augment = root, names, size, augment
        self.masks = {p.stem: p for p in (root / "masks").iterdir() if p.suffix.lower() in EXTS}
        self.images = {p.stem: p for p in (root / "images").iterdir() if p.suffix.lower() in EXTS}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.cvtColor(cv2.imread(str(self.images[name])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[name]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            # Colour/brightness jitter: endoscope processors and capsules differ a lot.
            img = np.clip(img.astype(np.float32) * random.uniform(0.75, 1.25) + random.uniform(-20, 20), 0, 255)
        x = (np.ascontiguousarray(img).astype(np.float32) / 255.0 - MEAN) / STD
        y = (np.ascontiguousarray(mask) > 127).astype(np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1)).float(), torch.from_numpy(y)[None]


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Compact U-Net. Swap for a pretrained-encoder model (e.g. PraNet, Polyp-PVT) for production."""

    def __init__(self, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList([block(3, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.decs = nn.ModuleList([block(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.decs):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)  # logits


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((2, 3)) + y.sum((2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices, ious = [], []
    for x, y in loader:
        p = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu()
        inter = (p * y).sum((1, 2, 3))
        union = ((p + y) > 0).float().sum((1, 2, 3))
        dices += ((2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).tolist()
        ious += ((inter + 1) / (union + 1)).tolist()
    return float(np.mean(dices)), float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, help='JSON {"train": [names], "val": [names]} split by patient')
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", type=Path, help="weights to start from (e.g. synthetic pre-training)")
    ap.add_argument("--out", type=Path, default=Path("models/unet.pt"))
    ap.add_argument("--export", type=Path, help="write ONNX model + JSON sidecar for SecondLook")
    ap.add_argument("--name", default="unet-polyp-seg")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.split_file:
        split = json.loads(args.split_file.read_text())
        train_names, val_names = split["train"], split["val"]
    else:
        print("WARNING: no --split-file; using a random frame-level split. Scores will be optimistic.")
        names = sorted(p.stem for p in (args.data / "masks").iterdir() if p.suffix.lower() in EXTS)
        random.shuffle(names)
        cut = max(1, int(0.8 * len(names)))
        train_names, val_names = names[:cut], names[cut:] or names[:1]

    train = DataLoader(PolypDataset(args.data, train_names, args.size, True), args.batch, shuffle=True, num_workers=2)
    val = DataLoader(PolypDataset(args.data, val_names, args.size, False), args.batch, num_workers=2)

    model = UNet().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in train:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        dice, iou = evaluate(model, val, device)
        print(f"epoch {epoch + 1}/{args.epochs} loss {np.mean(losses):.4f} val dice {dice:.4f} iou {iou:.4f}")
        if dice > best:
            best = dice
            torch.save(model.state_dict(), args.out)

    if args.export:
        model.load_state_dict(torch.load(args.out, map_location="cpu"))
        model.cpu().eval()
        args.export.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, torch.zeros(1, 3, args.size, args.size), str(args.export),
                          input_names=["image"], output_names=["logits"], opset_version=17)
        sidecar = {
            "name": args.name, "version": args.version, "input_size": args.size,
            "mean": MEAN, "std": STD, "threshold": 0.5, "label": "polyp", "output": "logits",
            "training_data": str(args.data), "val_dice": best,
        }
        args.export.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
        print(f"Exported {args.export} (val dice {best:.4f})")


if __name__ == "__main__":
    main()
