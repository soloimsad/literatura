from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from model_registry import get_model_spec


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


def roles_from_choice(choice: str) -> list[str]:
    if choice == "both":
        return ["tooth", "treatment"]
    return [choice]


def ensure_model_under_models(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    models_root = (PROJECT_ROOT / "models").resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"No existe el modelo local: {resolved}")
    if not str(resolved).lower().startswith(str(models_root).lower()):
        raise ValueError(f"El modelo debe estar dentro de {models_root}: {resolved}")
    return resolved


def role_model_path(role: str, args: argparse.Namespace) -> Path:
    if role == "tooth" and args.model:
        return ensure_model_under_models(Path(args.model))
    override = args.tooth_model if role == "tooth" else args.treatment_model
    if override:
        return ensure_model_under_models(Path(override))
    return ensure_model_under_models(get_model_spec(role).weights)


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


def classification_rows(result, role: str) -> list[dict]:
    probs = getattr(result, "probs", None)
    if probs is None:
        return []

    names = result.names or {}
    indices = list(getattr(probs, "top5", []) or [])
    confidences = list(getattr(probs, "top5conf", []) or [])
    rows = []

    for rank, (class_id, confidence) in enumerate(zip(indices, confidences), 1):
        if hasattr(confidence, "item"):
            confidence = confidence.item()
        rows.append(
            {
                "role": role,
                "raw_index": rank - 1,
                "instance_id": f"C{rank:02d}",
                "jaw": "",
                "class_id": int(class_id),
                "class_name": str(names.get(int(class_id), class_id)),
                "confidence": float(confidence),
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            }
        )
    return rows


def result_rows(result, role: str) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return classification_rows(result, role)

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    names = result.names or {}
    rows = []

    for raw_index, (box, confidence, class_id) in enumerate(zip(boxes, confs, classes)):
        x1, y1, x2, y2 = [float(value) for value in box]
        rows.append(
            {
                "role": role,
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

    if role != "tooth":
        for index, row in enumerate(sorted(rows, key=lambda item: (item["cy"], item["cx"])), 1):
            row["jaw"] = ""
            row["instance_id"] = f"D{index:02d}"
        return rows

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
    text_only_y = 24
    for row in rows:
        if row.get("x1") == "":
            label = f"{display_label(row['class_name'])} {row['confidence']:.2f}"
            cv2.putText(
                image,
                label,
                (12, text_only_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (46, 204, 113),
                2,
                cv2.LINE_AA,
            )
            text_only_y += 28
            continue

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


def save_report(rows: list[dict], image_path: Path, report_path: Path, output_image: Path) -> None:
    fields = [
        "role",
        "image",
        "instance_id",
        "jaw",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "output_image",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "role": row["role"],
                    "image": image_path.name,
                    "instance_id": row["instance_id"],
                    "jaw": row["jaw"],
                    "class_name": row["class_name"],
                    "confidence": f"{row['confidence']:.4f}",
                    "x1": f"{row['x1']:.1f}" if row["x1"] != "" else "",
                    "y1": f"{row['y1']:.1f}" if row["y1"] != "" else "",
                    "x2": f"{row['x2']:.1f}" if row["x2"] != "" else "",
                    "y2": f"{row['y2']:.1f}" if row["y2"] != "" else "",
                    "output_image": str(output_image),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta modelos locales de pieza dental y/o posible tratamiento sobre una radiografia."
    )
    parser.add_argument(
        "--model-role",
        choices=["tooth", "treatment", "both"],
        default="tooth",
        help="Modelo a ejecutar: pieza dental, tratamiento o ambos.",
    )
    parser.add_argument("--model", default=None, help="Compatibilidad: ruta dentro de models/ al modelo de pieza dental.")
    parser.add_argument(
        "--tooth-model",
        default=None,
        help="Ruta dentro de models/ al modelo de clasificacion de pieza dental.",
    )
    parser.add_argument(
        "--treatment-model",
        default=None,
        help="Ruta dentro de models/ al modelo de posible tratamiento.",
    )
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
    image_path = Path(args.image)
    output_root = Path(args.output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    device = 0 if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"

    combined_rows = []
    for role in roles_from_choice(args.model_role):
        spec = get_model_spec(role)
        model_path = role_model_path(role, args)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo para '{spec.display_name}': {model_path}. "
                "Ubica el .pt dentro de models/ o usa un argumento manual que apunte a models/."
            )

        output_dir = output_root / spec.output_dir
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
        rows = result_rows(result, role)

        output_image = output_dir / f"{image_path.stem}_{spec.output_dir}_labeled.jpg"
        output_report = output_dir / f"{image_path.stem}_{spec.output_dir}_report.csv"
        draw_segmented_view(result, rows, output_image)
        save_report(rows, image_path, output_report, output_image)
        combined_rows.extend(rows)

        print(f"Modelo ejecutado: {spec.display_name}")
        print("Modelo:", model_path)
        print("Imagen segmentada/rotulada:", output_image)
        print("Reporte CSV:", output_report)
        print("Predicciones:", len(rows))

    if len(roles_from_choice(args.model_role)) > 1:
        combined_report = output_root / f"{image_path.stem}_combined_report.csv"
        save_report(combined_rows, image_path, combined_report, output_root)
        print("Reporte combinado:", combined_report)


if __name__ == "__main__":
    main()
