"""Train the dental CNN segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    BoxMaskDataset,
    discover_records,
    filter_records_by_classes,
    limit_records,
    resolve_dataset_roots,
)
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", action="append", default=[], help="Dataset root path. Can be repeated.")
    parser.add_argument("--no-download", action="store_true", help="Do not download default Kaggle datasets.")
    parser.add_argument("--source", choices=["auto", "csv", "coco"], default="auto")
    parser.add_argument(
        "--include-class",
        action="append",
        default=[],
        help="Keep only this annotation class. Can be repeated, e.g. --include-class Caries.",
    )
    parser.add_argument("--mask-mode", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "dental_unet")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    roots = resolve_dataset_roots(args.data_root, download=not args.no_download)
    train_records, train_classes = discover_records(roots, "train", args.source)
    valid_records, valid_classes = discover_records(roots, "valid", args.source)
    train_records = filter_records_by_classes(train_records, args.include_class)
    valid_records = filter_records_by_classes(valid_records, args.include_class)
    if args.include_class:
        train_classes = sorted({box.class_name for rec in train_records for box in rec.boxes}, key=str.lower)
        valid_classes = sorted({box.class_name for rec in valid_records for box in rec.boxes}, key=str.lower)
    class_names = sorted(set(train_classes) | set(valid_classes), key=str.lower)

    if not train_records:
        raise SystemExit("No training records found. Check --data-root or run download_datasets.py.")
    if not valid_records:
        split_at = max(1, int(len(train_records) * 0.85))
        valid_records = train_records[split_at:]
        train_records = train_records[:split_at]

    train_records = limit_records(train_records, args.max_train_samples or None)
    valid_records = limit_records(valid_records, args.max_valid_samples or None)

    image_size = (args.image_width, args.image_height)
    train_ds = BoxMaskDataset(train_records, class_names, image_size, args.mask_mode, augment=True)
    valid_ds = BoxMaskDataset(valid_records, class_names, image_size, args.mask_mode, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.mask_mode, len(class_names), args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() if args.mask_mode == "binary" else nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history: list[dict[str, Any]] = []

    print(f"Device: {device}")
    print(f"Train records: {len(train_ds)} | Valid records: {len(valid_ds)}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, criterion, optimizer, device, args.mask_mode)
        valid_loss, metrics = run_eval_epoch(model, valid_loader, criterion, device, args.mask_mode)
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, **metrics}
        history.append(row)

        score = metrics.get("dice", metrics.get("pixel_accuracy", 0.0))
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} metrics={metrics}"
        )
        if score > best_score:
            best_score = score
            save_checkpoint(args.output_dir / "best_model.pt", model, args, class_names, best_score)

    save_checkpoint(args.output_dir / "last_model.pt", model, args, class_names, best_score)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {args.output_dir / 'best_model.pt'}")


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_mode: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = compute_loss(logits, masks, criterion, mask_mode)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


@torch.no_grad()
def run_eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    mask_mode: str,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    metric_accumulator: dict[str, list[float]] = {"dice": [], "iou": [], "pixel_accuracy": []}

    for images, masks in tqdm(loader, desc="valid", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = compute_loss(logits, masks, criterion, mask_mode)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        if mask_mode == "binary":
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            target = masks.float()
            metric_accumulator["dice"].append(float(binary_dice(preds, target).item()))
            metric_accumulator["iou"].append(float(binary_iou(preds, target).item()))
        else:
            preds = torch.argmax(logits, dim=1)
            metric_accumulator["pixel_accuracy"].append(float((preds == masks).float().mean().item()))

    metrics = {
        key: float(np.mean(values))
        for key, values in metric_accumulator.items()
        if values
    }
    return total_loss / max(1, total_items), metrics


def compute_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    criterion: torch.nn.Module,
    mask_mode: str,
) -> torch.Tensor:
    if mask_mode == "binary":
        bce = criterion(logits, masks.float())
        dice = 1.0 - binary_dice(torch.sigmoid(logits), masks.float())
        return bce + dice
    return criterion(logits, masks.long())


def binary_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    return ((2 * intersection + eps) / (denominator + eps)).mean()


def binary_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    class_names: list[str],
    best_score: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "mask_mode": args.mask_mode,
            "image_size": (args.image_width, args.image_height),
            "base_channels": args.base_channels,
            "best_score": best_score,
            "project": "proyecto_dental_cnn",
        },
        path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
