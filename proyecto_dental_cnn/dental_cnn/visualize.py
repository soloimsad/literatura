"""Visualization helpers for segmented dental radiographs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .postprocess import Instance


PALETTE = np.array(
    [
        [230, 57, 70],
        [42, 157, 143],
        [244, 162, 97],
        [69, 123, 157],
        [131, 56, 236],
        [255, 183, 3],
        [6, 214, 160],
        [239, 71, 111],
        [17, 138, 178],
        [255, 209, 102],
    ],
    dtype=np.uint8,
)


def draw_instances(image_path: Path, instances: list[Instance], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    overlay = image.copy()
    for idx, inst in enumerate(instances):
        color = tuple(int(v) for v in PALETTE[idx % len(PALETTE)])
        color_bgr = (color[2], color[1], color[0])
        overlay[inst.mask > 0] = (
            0.55 * overlay[inst.mask > 0] + 0.45 * np.array(color_bgr, dtype=np.float32)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_bgr, 2)
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 1)
        text = inst.label if inst.class_name == "Tooth" else f"{inst.label}:{inst.class_name}"
        put_label(overlay, text, x1, max(0, y1 - 6), color_bgr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def put_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(h + baseline + 2, y)
    x = max(0, min(x, image.shape[1] - w - 2))
    cv2.rectangle(image, (x, y - h - baseline - 3), (x + w + 4, y + baseline + 1), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, y - 2), font, scale, color, thickness, cv2.LINE_AA)


def save_instances_json(instances: list[Instance], path: Path) -> None:
    payload = [
        {
            "label": inst.label,
            "class_name": inst.class_name,
            "bbox": list(inst.bbox),
            "area": inst.area,
            "score": round(inst.score, 4),
        }
        for inst in instances
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
