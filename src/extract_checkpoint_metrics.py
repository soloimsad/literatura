from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from model_registry import MODEL_REGISTRY, PROJECT_ROOT


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 5)
    return value


def last_value(values: Any) -> Any:
    if isinstance(values, list) and values:
        return values[-1]
    return values


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint inesperado en {path}: {type(checkpoint).__name__}")
    return checkpoint


def extract_model_names(checkpoint: dict[str, Any]) -> dict[int, str]:
    model = checkpoint.get("ema") or checkpoint.get("model")
    names = getattr(model, "names", {}) if model is not None else {}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def extract_metrics(role: str, weights: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(weights)
    metrics = checkpoint.get("train_metrics") or {}
    train_results = checkpoint.get("train_results") or {}
    train_args = checkpoint.get("train_args") or {}
    names = extract_model_names(checkpoint)

    precision_box = metrics.get("metrics/precision(B)")
    recall_box = metrics.get("metrics/recall(B)")
    precision_mask = metrics.get("metrics/precision(M)")
    recall_mask = metrics.get("metrics/recall(M)")

    result = {
        "role": role,
        "weights": str(weights),
        "checkpoint_date": checkpoint.get("date"),
        "ultralytics_version": checkpoint.get("version"),
        "task": train_args.get("task"),
        "base_model": train_args.get("model"),
        "original_data_yaml": train_args.get("data"),
        "epochs_configured": train_args.get("epochs"),
        "epochs_recorded": len(train_results.get("epoch", [])) if isinstance(train_results, dict) else None,
        "imgsz": train_args.get("imgsz"),
        "batch": train_args.get("batch"),
        "classes": len(names),
        "class_names": names,
        "box": {
            "precision": precision_box,
            "recall": recall_box,
            "f1": f1_score(precision_box, recall_box),
            "map50": metrics.get("metrics/mAP50(B)"),
            "map50_95": metrics.get("metrics/mAP50-95(B)"),
        },
        "mask": {
            "precision": precision_mask,
            "recall": recall_mask,
            "f1": f1_score(precision_mask, recall_mask),
            "map50": metrics.get("metrics/mAP50(M)"),
            "map50_95": metrics.get("metrics/mAP50-95(M)"),
        },
        "validation_losses": {
            "box_loss": metrics.get("val/box_loss"),
            "seg_loss": metrics.get("val/seg_loss"),
            "cls_loss": metrics.get("val/cls_loss"),
            "dfl_loss": metrics.get("val/dfl_loss"),
        },
        "fitness": metrics.get("fitness", checkpoint.get("best_fitness")),
        "note": (
            "F1 se calcula desde precision y recall guardados en el checkpoint. "
            "DSC/Dice no viene guardado directamente; debe recalcularse con "
            "predicciones y mascaras ground truth del dataset de validacion."
        ),
    }
    return result


def markdown_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Metricas extraidas desde checkpoints",
        "",
        "Estas metricas fueron extraidas desde los archivos `.pt` locales. No son una nueva validacion; son los valores guardados dentro del checkpoint de entrenamiento.",
        "",
        "| Modelo | Clases | Precision box | Recall box | F1 box | mAP50 box | Precision mask | Recall mask | F1 mask | mAP50 mask |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        box = row["box"]
        mask = row["mask"]
        lines.append(
            "| {role} | {classes} | {pb} | {rb} | {fb} | {mb} | {pm} | {rm} | {fm} | {mm} |".format(
                role=row["role"],
                classes=row["classes"],
                pb=rounded(box["precision"]),
                rb=rounded(box["recall"]),
                fb=rounded(box["f1"]),
                mb=rounded(box["map50"]),
                pm=rounded(mask["precision"]),
                rm=rounded(mask["recall"]),
                fm=rounded(mask["f1"]),
                mm=rounded(mask["map50"]),
            )
        )

    lines.extend(
        [
            "",
            "## Detalle por modelo",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['role']}",
                "",
                f"- Pesos: `{row['weights']}`",
                f"- Fecha del checkpoint: `{row['checkpoint_date']}`",
                f"- Version Ultralytics: `{row['ultralytics_version']}`",
                f"- Modelo base declarado: `{row['base_model']}`",
                f"- Dataset original declarado: `{row['original_data_yaml']}`",
                f"- Epocas configuradas: `{row['epochs_configured']}`",
                f"- Epocas registradas: `{row['epochs_recorded']}`",
                f"- Imagen de entrenamiento: `{row['imgsz']}`",
                f"- Batch: `{row['batch']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Sobre DSC/Dice",
            "",
            "El checkpoint no guarda DSC/Dice como valor directo. Para obtener DSC se necesita ejecutar el modelo sobre un conjunto de validacion con mascaras ground truth compatibles y comparar pixel a pixel la mascara predicha contra la mascara real.",
            "",
            "Para el modelo de pieza dental, el checkpoint declara 54 clases tipo FDI/numara, mientras que el dataset local preparado en `data/_prepared_teeth_yolo_seg/data.yaml` tiene 32 clases numericas. Por eso no conviene recalcular metricas de pieza dental sobre ese dataset local como si fuera la validacion original.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae metricas guardadas dentro de checkpoints YOLO.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "metrics"),
        help="Directorio donde se escriben el JSON y el reporte Markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for role, spec in MODEL_REGISTRY.items():
        if not spec.weights.exists():
            print(f"No existe el peso para {role}: {spec.weights}")
            continue
        rows.append(extract_metrics(role, spec.weights))

    json_path = output_dir / "checkpoint_metrics.json"
    md_path = output_dir / "checkpoint_metrics.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_report(rows), encoding="utf-8")

    print("Metricas JSON:", json_path)
    print("Reporte Markdown:", md_path)


if __name__ == "__main__":
    main()
