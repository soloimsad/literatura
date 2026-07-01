from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from dental_xray_pipeline import (
    PREPARED_TEETH_DIR,
    PROJECT_ROOT,
    SEED,
    audit_segmentation_dataset,
    audit_teeth_dataset,
    ensure_model_under_models,
    find_segmentation_yolo_dir,
    find_teeth_supervisely_dir,
    make_local_yolo_yaml,
    prepare_teeth_supervisely_yolo,
    read_yaml,
    resolve_segmentation_dataset,
    resolve_teeth_dataset,
    seed_everything,
    write_audit_files,
)
from model_registry import get_model_spec


RUN_NAMES = {
    "tooth": "tooth_piece_classifier",
    "treatment": "treatment_detector",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena o reentrena los modelos locales YOLO-seg del proyecto."
    )
    parser.add_argument(
        "--model-role",
        required=True,
        choices=("tooth", "treatment", "both"),
        help="Modelo a entrenar: pieza dental, tratamiento/hallazgo o ambos.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Peso local de partida dentro de models/. Si se omite, usa el best.pt del rol.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu o indice de GPU. Ejemplo: --device 0",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre de la corrida dentro de models/. Solo se permite con un rol individual.",
    )
    parser.add_argument(
        "--run-suffix",
        default="retrain",
        help="Sufijo para no sobrescribir pesos existentes cuando no se entrega --run-name.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Solo audita y prepara el dataset; no ejecuta entrenamiento.",
    )
    parser.add_argument(
        "--allow-class-mismatch",
        action="store_true",
        help=(
            "Permite entrenar aunque el numero de clases del dataset no coincida "
            "con el modelo registrado. Util para experimentos, no para afirmar "
            "que reproduce el best.pt actual."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desactiva graficos generados por Ultralytics.",
    )
    args = parser.parse_args()

    if args.model_role == "both" and args.run_name:
        parser.error("--run-name solo se puede usar cuando --model-role es tooth o treatment.")

    return args


def resolve_device(value: str):
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if value.lower() == "cpu":
        return "cpu"
    try:
        return int(value)
    except ValueError:
        return value


def resolve_base_model(role: str, base_model: str | None) -> Path:
    if base_model:
        path = Path(base_model)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return ensure_model_under_models(path)
    return ensure_model_under_models(get_model_spec(role).weights)


def default_run_name(role: str, suffix: str) -> str:
    base_name = RUN_NAMES[role]
    clean_suffix = suffix.strip()
    if not clean_suffix:
        return base_name
    return f"{base_name}_{clean_suffix}"


def write_preparation_summary(work_dir: Path, preparation_summary: dict) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "preparation_summary.json").write_text(
        json.dumps(preparation_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def prepare_tooth_dataset(work_dir: Path) -> Path:
    dataset_root = resolve_teeth_dataset()
    supervisely_dir = find_teeth_supervisely_dir(dataset_root)
    audit = audit_teeth_dataset(dataset_root, supervisely_dir)
    write_audit_files(audit, work_dir)

    data_yaml, preparation_summary = prepare_teeth_supervisely_yolo(
        supervisely_dir,
        PREPARED_TEETH_DIR,
    )
    write_preparation_summary(work_dir, preparation_summary)
    return data_yaml


def prepare_treatment_dataset(work_dir: Path) -> Path:
    dataset_root = resolve_segmentation_dataset()
    yolo_dir = find_segmentation_yolo_dir(dataset_root)
    audit = audit_segmentation_dataset(dataset_root, yolo_dir)
    write_audit_files(audit, work_dir)

    data_yaml, preparation_summary = make_local_yolo_yaml(
        yolo_dir,
        work_dir / "prepared_yolo_seg" / "data.yaml",
    )
    write_preparation_summary(work_dir, preparation_summary)
    return data_yaml


def prepare_dataset(role: str, work_dir: Path) -> Path:
    if role == "tooth":
        return prepare_tooth_dataset(work_dir)
    if role == "treatment":
        return prepare_treatment_dataset(work_dir)
    raise ValueError(f"Rol no soportado: {role}")


def class_count(data_yaml: Path) -> int:
    names = read_yaml(data_yaml).get("names", {})
    return len(names)


def validate_class_count(role: str, data_yaml: Path, allow_mismatch: bool) -> None:
    dataset_classes = class_count(data_yaml)
    expected_classes = len(get_model_spec(role).expected_classes)
    if dataset_classes == expected_classes:
        return

    message = (
        f"El dataset preparado para '{role}' tiene {dataset_classes} clases, "
        f"pero el modelo registrado espera {expected_classes}. "
        "No conviene afirmar que este entrenamiento reproduce el best.pt actual."
    )
    if allow_mismatch:
        print("ADVERTENCIA:", message)
        return
    raise ValueError(
        message
        + " Usa --allow-class-mismatch solo si quieres entrenar un experimento nuevo "
        "con otro conjunto de clases."
    )


def train_role(role: str, args: argparse.Namespace) -> Path | None:
    seed_everything()

    work_dir = PROJECT_ROOT / "results" / "training" / role
    data_yaml = prepare_dataset(role, work_dir)
    validate_class_count(role, data_yaml, args.allow_class_mismatch)

    print("Rol:", role)
    print("YAML de entrenamiento:", data_yaml)

    if args.analyze_only:
        print("ANALYZE_ONLY: dataset preparado; entrenamiento omitido.")
        return None

    base_model = resolve_base_model(role, args.base_model)
    run_name = args.run_name or default_run_name(role, args.run_suffix)
    device = resolve_device(args.device)

    print("Modelo base local:", base_model)
    print("Salida de entrenamiento:", PROJECT_ROOT / "models" / run_name)
    print("Dispositivo:", device)

    model = YOLO(str(base_model))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=device,
        project=str(PROJECT_ROOT / "models"),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        seed=SEED,
        plots=not args.no_plots,
        amp=torch.cuda.is_available() and device != "cpu",
    )

    best_model = PROJECT_ROOT / "models" / run_name / "weights" / "best.pt"
    print("Peso entrenado:", best_model)
    return best_model


def main() -> None:
    args = parse_args()
    roles = ("tooth", "treatment") if args.model_role == "both" else (args.model_role,)

    trained = []
    for role in roles:
        best_model = train_role(role, args)
        if best_model:
            trained.append(str(best_model))

    if trained:
        print("Modelos generados:")
        for path in trained:
            print("-", path)


if __name__ == "__main__":
    main()
