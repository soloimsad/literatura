from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "teeth_segmentation" / "Teeth Segmentation JSON" / "d2"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "data" / "_prepared_tooth_piece_classifier"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "tooth_piece_cnn_classifier"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEED = 42


@dataclass(frozen=True)
class CropRecord:
    split: str
    image: Path
    annotation: Path
    crop: Path
    class_name: str
    class_id: int
    object_id: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def find_supervisely_dir(source_dir: Path) -> Path:
    if (source_dir / "ann").exists() and (source_dir / "img").exists():
        return source_dir

    preferred = [
        source_dir / "Teeth Segmentation JSON" / "d2",
        source_dir / "d2",
    ]
    for candidate in preferred:
        if (candidate / "ann").exists() and (candidate / "img").exists():
            return candidate

    for ann_dir in sorted(source_dir.rglob("ann")):
        candidate = ann_dir.parent
        if (candidate / "img").exists():
            return candidate

    raise FileNotFoundError(
        "No se encontro un dataset Supervisely con carpetas ann/ e img/ en "
        f"{source_dir}"
    )


def sorted_class_names(annotation_paths: list[Path]) -> list[str]:
    classes = set()
    for ann_path in annotation_paths:
        data = read_json(ann_path)
        for obj in data.get("objects", []):
            if obj.get("geometryType") != "polygon":
                continue
            points = obj.get("points", {}).get("exterior", [])
            class_name = str(obj.get("classTitle", "")).strip()
            if class_name and len(points) >= 3:
                classes.add(class_name)

    def key(value: str) -> tuple[int, str]:
        return (int(value), value) if value.isdigit() else (10_000, value)

    return sorted(classes, key=key)


def split_annotation_files(
    annotation_paths: list[Path],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    shuffled = list(annotation_paths)
    random.Random(seed).shuffle(shuffled)

    train_count = max(1, int(len(shuffled) * train_ratio))
    valid_count = max(1, int(len(shuffled) * valid_ratio))
    if train_count + valid_count >= len(shuffled):
        valid_count = max(1, len(shuffled) - train_count - 1)

    return {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + valid_count],
        "test": shuffled[train_count + valid_count :],
    }


def crop_polygon(
    image: np.ndarray,
    points: list,
    padding: int,
    mask_background: bool,
) -> np.ndarray | None:
    polygon = np.array(points, dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return None

    height, width = image.shape[:2]
    x, y, w, h = cv2.boundingRect(polygon)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2].copy()
    if not mask_background:
        return crop

    relative_polygon = polygon.copy()
    relative_polygon[:, 0] -= x1
    relative_polygon[:, 1] -= y1
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [relative_polygon], 255)
    return cv2.bitwise_and(crop, crop, mask=mask)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def prepare_crops(
    source_dir: Path,
    prepared_dir: Path,
    image_size: int,
    padding: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    force: bool,
    mask_background: bool,
    limit_images: int | None,
) -> dict:
    supervisely_dir = find_supervisely_dir(source_dir)
    ann_dir = supervisely_dir / "ann"
    img_dir = supervisely_dir / "img"
    annotation_paths = sorted(ann_dir.glob("*.json"))
    if limit_images:
        annotation_paths = annotation_paths[:limit_images]
    if not annotation_paths:
        raise FileNotFoundError(f"No hay anotaciones JSON en {ann_dir}")

    manifest_path = prepared_dir / "manifest.csv"
    if manifest_path.exists() and not force:
        return read_preparation_summary(prepared_dir)

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted_class_names(annotation_paths)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    splits = split_annotation_files(annotation_paths, train_ratio, valid_ratio, seed)

    records: list[CropRecord] = []
    skipped = Counter()
    split_counts: dict[str, Counter] = {split: Counter() for split in splits}

    for split, split_annotations in splits.items():
        for ann_path in split_annotations:
            data = read_json(ann_path)
            image_name = ann_path.name[:-5] if ann_path.name.endswith(".json") else ann_path.stem
            image_path = img_dir / image_name
            if not image_path.exists():
                skipped["missing_image"] += 1
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                skipped["unreadable_image"] += 1
                continue

            for obj_index, obj in enumerate(data.get("objects", [])):
                class_name = str(obj.get("classTitle", "")).strip()
                points = obj.get("points", {}).get("exterior", [])
                if obj.get("geometryType") != "polygon" or class_name not in class_to_id:
                    skipped["invalid_object"] += 1
                    continue
                if len(points) < 3:
                    skipped["invalid_polygon"] += 1
                    continue

                crop = crop_polygon(image, points, padding, mask_background)
                if crop is None or crop.size == 0:
                    skipped["empty_crop"] += 1
                    continue
                crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)

                class_dir = prepared_dir / split / safe_name(class_name)
                class_dir.mkdir(parents=True, exist_ok=True)
                crop_name = f"{Path(image_name).stem}_obj{obj_index:03d}.png"
                crop_path = class_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                class_id = class_to_id[class_name]
                split_counts[split][class_name] += 1
                records.append(
                    CropRecord(
                        split=split,
                        image=image_path,
                        annotation=ann_path,
                        crop=crop_path,
                        class_name=class_name,
                        class_id=class_id,
                        object_id=str(obj.get("id", obj_index)),
                    )
                )

    write_manifest(manifest_path, records)
    summary = {
        "source_dir": str(supervisely_dir),
        "prepared_dir": str(prepared_dir),
        "image_size": image_size,
        "padding": padding,
        "mask_background": mask_background,
        "annotation_files": len(annotation_paths),
        "classes": class_names,
        "class_to_id": class_to_id,
        "splits": {
            split: {
                "images": len(split_annotations),
                "crops": sum(split_counts[split].values()),
                "class_counts": dict(sorted(split_counts[split].items(), key=lambda item: class_to_id[item[0]])),
            }
            for split, split_annotations in splits.items()
        },
        "skipped": dict(skipped),
    }
    (prepared_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def write_manifest(path: Path, records: list[CropRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "crop", "class_name", "class_id", "image", "annotation", "object_id"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record.split,
                    "crop": record.crop,
                    "class_name": record.class_name,
                    "class_id": record.class_id,
                    "image": record.image,
                    "annotation": record.annotation,
                    "object_id": record.object_id,
                }
            )


def read_preparation_summary(prepared_dir: Path) -> dict:
    summary_path = prepared_dir / "preparation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Existe {prepared_dir / 'manifest.csv'}, pero falta {summary_path}. "
            "Ejecuta con --force-prepare."
        )
    return read_json(summary_path)


class ToothCropDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, image_size: int, augment: bool = False) -> None:
        self.rows = []
        self.image_size = image_size
        self.augment = augment
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["split"] == split:
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No hay muestras para split={split} en {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = cv2.imread(row["crop"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(row["crop"])
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = self.apply_augmentation(image)
        image = image.astype(np.float32) / 255.0
        mean = float(image.mean())
        std = float(image.std())
        if std < 1e-6:
            std = 1.0
        image = (image - mean) / std
        tensor = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(int(row["class_id"]), dtype=torch.long)
        return tensor, label

    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        if not self.augment:
            return image
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-0.05, 0.05) * 255.0
        adjusted = image.astype(np.float32) * alpha + beta
        return np.clip(adjusted, 0, 255).astype(np.uint8)


class ToothPieceCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self.block(1, 32),
            nn.MaxPool2d(2),
            self.block(32, 64),
            nn.MaxPool2d(2),
            self.block(64, 128),
            nn.MaxPool2d(2),
            self.block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def class_weights_from_manifest(manifest_path: Path, num_classes: int) -> torch.Tensor:
    counts = Counter()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] == "train":
                counts[int(row["class_id"])] += 1
    total = sum(counts.values())
    weights = []
    for class_id in range(num_classes):
        count = max(1, counts[class_id])
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0
    precision_values = []
    recall_values = []
    f1_values = []
    for class_id in range(confusion.shape[0]):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "samples": total,
        "correct": correct,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, dict, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += float(loss.item()) * labels.size(0)
            sample_count += labels.size(0)
            preds = torch.argmax(logits, dim=1)
            for truth, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[int(truth), int(pred)] += 1
    metrics = metrics_from_confusion(confusion)
    return loss_sum / max(1, sample_count), metrics, confusion


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path: Path, confusion: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for class_name, row in zip(class_names, confusion):
            writer.writerow([class_name, *row.tolist()])


def train_model(args: argparse.Namespace, summary: dict) -> None:
    prepared_dir = Path(summary["prepared_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(summary["classes"])
    num_classes = len(class_names)

    manifest_path = prepared_dir / "manifest.csv"
    train_dataset = ToothCropDataset(manifest_path, "train", args.image_size, augment=True)
    valid_dataset = ToothCropDataset(manifest_path, "valid", args.image_size, augment=False)
    test_dataset = ToothCropDataset(manifest_path, "test", args.image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ToothPieceCNN(num_classes).to(device)
    weights = class_weights_from_manifest(manifest_path, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )

    best_f1 = -math.inf
    history = []
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * labels.size(0)
            train_samples += labels.size(0)

        train_loss = train_loss_sum / max(1, train_samples)
        valid_loss, valid_metrics, valid_confusion = evaluate(
            model, valid_loader, criterion, device, num_classes
        )
        scheduler.step(valid_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "valid_loss": round(valid_loss, 6),
            "valid_accuracy": round(valid_metrics["accuracy"], 6),
            "valid_macro_precision": round(valid_metrics["macro_precision"], 6),
            "valid_macro_recall": round(valid_metrics["macro_recall"], 6),
            "valid_macro_f1": round(valid_metrics["macro_f1"], 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"loss={train_loss:.4f} val_f1={valid_metrics['macro_f1']:.4f} "
            f"val_acc={valid_metrics['accuracy']:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": args.image_size,
            "epoch": epoch,
            "valid_metrics": valid_metrics,
            "args": vars(args),
            "architecture": "ToothPieceCNN",
        }
        torch.save(checkpoint, last_path)
        if valid_metrics["macro_f1"] > best_f1:
            best_f1 = valid_metrics["macro_f1"]
            torch.save(checkpoint, best_path)
            write_confusion_matrix(output_dir / "valid_confusion_matrix.csv", valid_confusion, class_names)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_metrics, test_confusion = evaluate(model, test_loader, criterion, device, num_classes)
    test_metrics["loss"] = test_loss

    write_history(output_dir / "training_history.csv", history)
    write_confusion_matrix(output_dir / "test_confusion_matrix.csv", test_confusion, class_names)
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics = {
        "best_valid_macro_f1": best_f1,
        "test": test_metrics,
        "prepared_summary": summary,
        "model": {
            "architecture": "ToothPieceCNN",
            "input_channels": 1,
            "image_size": args.image_size,
            "classes": num_classes,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(output_dir / "README.md", metrics, class_names)
    print("Modelo entrenado:", best_path)
    print("Metricas:", output_dir / "metrics.json")


def write_report(path: Path, metrics: dict, class_names: list[str]) -> None:
    test = metrics["test"]
    lines = [
        "# Clasificador CNN de piezas dentales",
        "",
        "Modelo deep learning entrenado desde recortes generados con poligonos del dataset Supervisely.",
        "",
        "## Resultado principal",
        "",
        f"- Best valid macro F1: `{metrics['best_valid_macro_f1']:.4f}`",
        f"- Test accuracy: `{test['accuracy']:.4f}`",
        f"- Test macro precision: `{test['macro_precision']:.4f}`",
        f"- Test macro recall: `{test['macro_recall']:.4f}`",
        f"- Test macro F1: `{test['macro_f1']:.4f}`",
        "",
        "## Archivos",
        "",
        "- `best.pt`: mejor checkpoint segun macro F1 de validacion.",
        "- `last.pt`: ultimo checkpoint entrenado.",
        "- `metrics.json`: metricas finales.",
        "- `training_history.csv`: perdida y metricas por epoca.",
        "- `valid_confusion_matrix.csv`: matriz de confusion de validacion.",
        "- `test_confusion_matrix.csv`: matriz de confusion de prueba.",
        "- `class_names.json`: clases del modelo.",
        "",
        "## Clases",
        "",
    ]
    for index, class_name in enumerate(class_names):
        lines.append(f"- {index}: {class_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara recortes por poligono y entrena una CNN para clasificar piezas dentales."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--no-mask-background", action="store_true")
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Limita la preparacion a N radiografias. Util solo para pruebas rapidas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    prepared_dir = Path(args.prepared_dir)
    if args.skip_prepare:
        summary = read_preparation_summary(prepared_dir)
    else:
        summary = prepare_crops(
            source_dir=Path(args.source_dir),
            prepared_dir=prepared_dir,
            image_size=args.image_size,
            padding=args.padding,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            force=args.force_prepare,
            mask_background=not args.no_mask_background,
            limit_images=args.limit_images,
        )

    print("Dataset preparado:", summary["prepared_dir"])
    print("Clases:", len(summary["classes"]))
    for split, info in summary["splits"].items():
        print(f"{split}: {info['crops']} recortes desde {info['images']} radiografias")

    if args.prepare_only:
        return

    train_model(args, summary)


if __name__ == "__main__":
    main()
