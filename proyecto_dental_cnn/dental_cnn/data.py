"""Dataset discovery and mask generation for panoramic dental radiographs.

The Kaggle datasets used here provide bounding boxes rather than pixel-perfect
tooth masks. Following the project papers, this code turns those annotations
into weak semantic masks for CNN training, then the prediction path refines the
output with image processing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .download_datasets import DATASET_SLUGS, download_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class AnnotationBox:
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    width: int
    height: int
    boxes: tuple[AnnotationBox, ...]
    source: str


def resolve_dataset_roots(paths: Iterable[str] | None, download: bool = True) -> list[Path]:
    roots: list[Path] = []
    if paths:
        roots.extend(Path(path).expanduser().resolve() for path in paths)
    elif download:
        roots.extend(download_dataset(slug) for slug in DATASET_SLUGS)

    missing = [str(path) for path in roots if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset roots not found: {missing}")
    return roots


def discover_records(
    roots: Iterable[Path],
    split: str,
    source_kind: str = "auto",
) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    class_names: set[str] = set()

    for root in roots:
        if source_kind in {"auto", "csv"}:
            csv_records = _load_roboflow_csv_records(root, split)
            records.extend(csv_records)
            class_names.update(box.class_name for rec in csv_records for box in rec.boxes)

        if source_kind in {"auto", "coco"}:
            coco_records = _load_coco_records(root, split)
            records.extend(coco_records)
            class_names.update(box.class_name for rec in coco_records for box in rec.boxes)

    records = [rec for rec in records if rec.boxes]
    return records, sorted(class_names, key=str.lower)


def normalize_class_name(name: str) -> str:
    cleaned = " ".join(str(name).replace("_", " ").split()).strip()
    aliases = {
        "Cavity": "Caries",
        "Fillings": "Filling",
        "Impacted Tooth": "impacted tooth",
        "croen": "Crown",
    }
    return aliases.get(cleaned, cleaned)


def filter_records_by_classes(records: list[ImageRecord], include: Iterable[str]) -> list[ImageRecord]:
    include_set = {normalize_class_name(name) for name in include}
    if not include_set:
        return records
    filtered: list[ImageRecord] = []
    for record in records:
        boxes = tuple(box for box in record.boxes if box.class_name in include_set)
        if boxes:
            filtered.append(
                ImageRecord(
                    image_path=record.image_path,
                    width=record.width,
                    height=record.height,
                    boxes=boxes,
                    source=record.source,
                )
            )
    return filtered


def _load_roboflow_csv_records(root: Path, split: str) -> list[ImageRecord]:
    csv_path = root / split / "_annotations.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    required = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected CSV columns in {csv_path}: {list(df.columns)}")

    records: list[ImageRecord] = []
    for filename, group in df.groupby("filename", sort=False):
        image_path = root / split / str(filename)
        if not image_path.exists():
            continue
        boxes = tuple(
            AnnotationBox(
                class_name=normalize_class_name(str(row["class"])),
                x1=float(row["xmin"]),
                y1=float(row["ymin"]),
                x2=float(row["xmax"]),
                y2=float(row["ymax"]),
            )
            for _, row in group.iterrows()
        )
        records.append(
            ImageRecord(
                image_path=image_path,
                width=int(group.iloc[0]["width"]),
                height=int(group.iloc[0]["height"]),
                boxes=boxes,
                source=f"roboflow-csv:{split}",
            )
        )
    return records


def _load_coco_records(root: Path, split: str) -> list[ImageRecord]:
    annotation_path = root / "COCO" / "COCO" / "annotations" / f"{split}_coco.json"
    image_dir = root / "COCO" / "COCO" / split
    if not annotation_path.exists() or not image_dir.exists():
        return []

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        int(cat["id"]): normalize_class_name(str(cat["name"]))
        for cat in data.get("categories", [])
    }
    images = {int(img["id"]): img for img in data.get("images", [])}
    grouped: dict[int, list[AnnotationBox]] = {image_id: [] for image_id in images}

    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id not in grouped or category_id not in categories:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        grouped[image_id].append(
            AnnotationBox(
                class_name=categories[category_id],
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h,
            )
        )

    records: list[ImageRecord] = []
    for image_id, image in images.items():
        boxes = tuple(grouped.get(image_id, []))
        image_path = image_dir / image["file_name"]
        if not image_path.exists() or not boxes:
            continue
        records.append(
            ImageRecord(
                image_path=image_path,
                width=int(image["width"]),
                height=int(image["height"]),
                boxes=boxes,
                source=f"coco:{split}",
            )
        )
    return records


class BoxMaskDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        class_names: list[str],
        image_size: tuple[int, int] = (256, 128),
        mask_mode: str = "multiclass",
        augment: bool = False,
    ) -> None:
        if mask_mode not in {"binary", "multiclass"}:
            raise ValueError("mask_mode must be 'binary' or 'multiclass'")
        self.records = records
        self.class_names = class_names
        self.class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
        self.image_size = image_size
        self.mask_mode = mask_mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = load_grayscale(record.image_path)
        mask = self._boxes_to_mask(record)

        image, mask = resize_pair(image, mask, self.image_size)
        if self.augment and np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))

        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image_tensor = torch.from_numpy(image[None, :, :])

        if self.mask_mode == "binary":
            mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)[None, :, :])
        else:
            mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor

    def _boxes_to_mask(self, record: ImageRecord) -> np.ndarray:
        mask = np.zeros((record.height, record.width), dtype=np.uint8)
        for box in record.boxes:
            x1 = int(np.clip(round(box.x1), 0, record.width - 1))
            y1 = int(np.clip(round(box.y1), 0, record.height - 1))
            x2 = int(np.clip(round(box.x2), x1 + 1, record.width))
            y2 = int(np.clip(round(box.y2), y1 + 1, record.height))
            value = 1 if self.mask_mode == "binary" else self.class_to_id[box.class_name]
            cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), int(value), thickness=-1)
        return mask


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized


def limit_records(records: list[ImageRecord], max_samples: int | None) -> list[ImageRecord]:
    if max_samples is None or max_samples <= 0:
        return records
    return records[:max_samples]
