from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PALETTE = [
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (231, 76, 60),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (149, 165, 166),
]


def display_label(class_name: str) -> str:
    match = re.search(r"\b\d{2}\b", class_name)
    if match:
        return match.group(0)
    return class_name[:18]


def jaw_split(rows: list[dict], image_height: int) -> float:
    if len(rows) < 4:
        return image_height / 2

    centers = sorted(row["cy"] for row in rows)
    gaps = [(centers[index + 1] - centers[index], index) for index in range(len(centers) - 1)]
    largest_gap, gap_index = max(gaps, key=lambda item: item[0])
    if largest_gap > image_height * 0.08:
        return (centers[gap_index] + centers[gap_index + 1]) / 2
    return image_height / 2


def result_rows(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    names = result.names or {}
    rows = []

    for raw_index, (box, confidence, class_id) in enumerate(zip(boxes, confs, classes)):
        x1, y1, x2, y2 = [float(value) for value in box]
        rows.append(
            {
                "raw_index": raw_index,
                "class_id": int(class_id),
                "class_name": str(names.get(int(class_id), class_id)),
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
            }
        )

    split_y = jaw_split(rows, result.orig_shape[0])
    for row in rows:
        row["jaw"] = "upper" if row["cy"] < split_y else "lower"

    ordered_rows = []
    for jaw in ("upper", "lower"):
        jaw_rows = sorted((row for row in rows if row["jaw"] == jaw), key=lambda item: item["cx"])
        for index, row in enumerate(jaw_rows, 1):
            prefix = "U" if jaw == "upper" else "L"
            row["instance_id"] = f"{prefix}{index:02d}"
            ordered_rows.append(row)
    return ordered_rows


def draw_segmented_view(result, rows: list[dict], output_path: Path) -> None:
    image = result.orig_img.copy()
    overlay = image.copy()

    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.detach().cpu().numpy()
        height, width = image.shape[:2]
        for row in rows:
            if row["raw_index"] >= len(masks):
                continue
            mask = masks[row["raw_index"]]
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            color = PALETTE[row["class_id"] % len(PALETTE)]
            overlay[mask > 0.5] = color

    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    for row in rows:
        x1, y1, x2, y2 = [int(round(row[key])) for key in ("x1", "y1", "x2", "y2")]
        color = PALETTE[row["class_id"] % len(PALETTE)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = display_label(row["class_name"])
        text_scale = 0.45
        thickness = 1
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
        text_x = max(0, min(x1, image.shape[1] - text_size[0] - 4))
        text_y = max(text_size[1] + 4, y1 - 4)
        cv2.rectangle(
            image,
            (text_x - 2, text_y - text_size[1] - baseline - 2),
            (text_x + text_size[0] + 2, text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (text_x, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def save_report(rows: list[dict], image_path: Path, report_path: Path) -> None:
    fields = ["image", "instance_id", "jaw", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": image_path.name,
                    "instance_id": row["instance_id"],
                    "jaw": row["jaw"],
                    "class_name": row["class_name"],
                    "confidence": f"{row['confidence']:.4f}",
                    "x1": f"{row['x1']:.1f}",
                    "y1": f"{row['y1']:.1f}",
                    "x2": f"{row['x2']:.1f}",
                    "y2": f"{row['y2']:.1f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera una radiografia dental segmentada y rotulada usando un modelo YOLO-seg entrenado."
    )
    parser.add_argument("--model", required=True, help="Ruta al modelo .pt de segmentacion.")
    parser.add_argument("--image", required=True, help="Ruta a la radiografia de entrada.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "inference"),
        help="Carpeta de salida.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Tamano de inferencia; 1280 recomendado para FDI.")
    parser.add_argument("--conf", type=float, default=0.10, help="Confianza minima.")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU para NMS.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' o indice GPU como '0'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    device = 0 if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=False,
    )
    result = results[0]
    rows = result_rows(result)

    output_image = output_dir / f"{image_path.stem}_segmented_labeled.jpg"
    output_report = output_dir / f"{image_path.stem}_report.csv"
    draw_segmented_view(result, rows, output_image)
    save_report(rows, image_path, output_report)

    print("Imagen segmentada/rotulada:", output_image)
    print("Reporte CSV:", output_report)
    print("Detecciones:", len(rows))


if __name__ == "__main__":
    main()
