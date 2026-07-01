from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


SEGMENTATION_DATASET_NAME = "dental-disease-panoramic-detection-dataset"
TEETH_DATASET_NAME = "Teeth Segmentation"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = Path(os.getenv("LOCAL_DATA_DIR", str(PROJECT_ROOT / "data")))
LOCAL_SEGMENTATION_DIR = LOCAL_DATA_DIR / SEGMENTATION_DATASET_NAME
LOCAL_TEETH_DIR_CANDIDATES = [LOCAL_DATA_DIR / "teeth_segmentation", LOCAL_DATA_DIR / TEETH_DATASET_NAME]
LOCAL_TEETH_DIR = next((path for path in LOCAL_TEETH_DIR_CANDIDATES if path.exists()), LOCAL_TEETH_DIR_CANDIDATES[0])
PREPARED_TEETH_DIR = Path(os.getenv("PREPARED_TEETH_DIR", LOCAL_DATA_DIR / "_prepared_teeth_yolo_seg"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEED = 42
TEETH_CLASS_NAMES = {index - 1: str(index) for index in range(1, 33)}
DEFAULT_DENTAL_DISEASE_NAMES = {
    0: "Caries",
    1: "Crown",
    2: "Filling",
    3: "Implant",
    4: "Malaligned",
    5: "Mandibular Canal",
    6: "Missing teeth",
    7: "Periapical lesion",
    8: "Retained root",
    9: "Root Canal Treatment",
    10: "Root Piece",
    11: "Impacted tooth",
    12: "Maxillary sinus",
    13: "Bone Loss",
    14: "Fracture teeth",
    15: "Permanent Teeth",
    16: "Supra Eruption",
    17: "TAD",
    18: "Abutment",
    19: "Attrition",
    20: "Bone defect",
    21: "Gingival former",
    22: "Metal band",
    23: "Orthodontic brackets",
    24: "Permanent retainer",
    25: "Post-core",
    26: "Plating",
    27: "Wire",
    28: "Cyst",
    29: "Root resorption",
    30: "Primary teeth",
}


def display_label(class_name: str) -> str:
    match = re.search(r"\b\d{2}\b", class_name)
    if match:
        return match.group(0)
    return class_name[:18]


def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def ensure_package(module_name: str, package_name: str | None = None) -> None:
    try:
        __import__(module_name)
        return
    except ImportError:
        pass
    pip_install(package_name or module_name)


ensure_package("yaml", "pyyaml")
ensure_package("cv2", "opencv-python")
ensure_package("ultralytics")

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def project_dir() -> Path:
    default = PROJECT_ROOT / "results"
    path = Path(os.getenv("PROJECT_DIR", str(default)))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_segmentation_dataset() -> Path:
    if os.getenv("DENTAL_SEGMENTATION_ROOT"):
        root = Path(os.environ["DENTAL_SEGMENTATION_ROOT"])
        if root.exists():
            return root
        raise FileNotFoundError(f"No existe DENTAL_SEGMENTATION_ROOT: {root}")
    if LOCAL_SEGMENTATION_DIR.exists() and any(LOCAL_SEGMENTATION_DIR.iterdir()):
        return LOCAL_SEGMENTATION_DIR
    raise FileNotFoundError(
        "No se encontro el dataset local de segmentacion. Esperado en "
        f"{LOCAL_SEGMENTATION_DIR}. Tambien puedes definir DENTAL_SEGMENTATION_ROOT."
    )


def resolve_teeth_dataset() -> Path:
    if os.getenv("TEETH_SEGMENTATION_ROOT"):
        root = Path(os.environ["TEETH_SEGMENTATION_ROOT"])
        if root.exists():
            return root
        raise FileNotFoundError(f"No existe TEETH_SEGMENTATION_ROOT: {root}")
    if LOCAL_TEETH_DIR.exists() and any(LOCAL_TEETH_DIR.iterdir()):
        return LOCAL_TEETH_DIR
    raise FileNotFoundError(
        "No se encontro el dataset local de dientes. Esperado en "
        f"{LOCAL_TEETH_DIR}. Tambien puedes definir TEETH_SEGMENTATION_ROOT."
    )


def choose_training_dataset() -> tuple[str, Path]:
    requested = os.getenv("TRAIN_DATASET", "auto").strip().lower()
    teeth_aliases = {"auto", "teeth", "tooth", "dientes"}
    disease_aliases = {"auto", "disease", "findings", "hallazgos", "enfermedades"}

    if requested not in teeth_aliases | disease_aliases:
        raise ValueError("TRAIN_DATASET debe ser auto, teeth o disease.")

    if requested in teeth_aliases and (os.getenv("TEETH_SEGMENTATION_ROOT") or LOCAL_TEETH_DIR.exists()):
        return "teeth", resolve_teeth_dataset()

    if requested in disease_aliases:
        return "disease", resolve_segmentation_dataset()

    raise FileNotFoundError(
        "TRAIN_DATASET=teeth fue solicitado, pero no se encontro el dataset "
        f"en {LOCAL_TEETH_DIR}."
    )


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def names_to_dict(names: dict | list) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(key): str(value) for key, value in names.items()}


def load_class_names(yolo_dir: Path) -> dict[int, str]:
    yaml_path = yolo_dir / "data.yaml"
    if yaml_path.exists():
        data = read_yaml(yaml_path)
        if data.get("names"):
            return names_to_dict(data["names"])
    return DEFAULT_DENTAL_DISEASE_NAMES.copy()


def has_yolo_segmentation_splits(candidate: Path) -> bool:
    required = [
        candidate / "train" / "images",
        candidate / "train" / "labels",
        candidate / "valid" / "images",
        candidate / "valid" / "labels",
    ]
    return all(path.exists() for path in required)


def find_segmentation_yolo_dir(root: Path) -> Path:
    for yaml_path in sorted(root.rglob("data.yaml")):
        parent = yaml_path.parent
        if has_yolo_segmentation_splits(parent):
            return parent

    for train_dir in sorted(root.rglob("train")):
        if not train_dir.is_dir():
            continue
        candidate = train_dir.parent
        if has_yolo_segmentation_splits(candidate):
            return candidate

    raise FileNotFoundError(
        "No se encontro el dataset YOLO segmentado dentro de "
        f"{root}. Revisa que existan las carpetas train/valid/test con images/labels."
    )


def has_teeth_supervisely_dirs(candidate: Path) -> bool:
    return (candidate / "ann").exists() and (candidate / "img").exists()


def find_teeth_supervisely_dir(root: Path) -> Path:
    preferred = [
        root / "Teeth Segmentation JSON" / "d2",
        root / "d2",
        root,
    ]
    for candidate in preferred:
        if has_teeth_supervisely_dirs(candidate):
            return candidate

    for ann_dir in sorted(root.rglob("ann")):
        if "__MACOSX" in ann_dir.parts:
            continue
        candidate = ann_dir.parent
        if has_teeth_supervisely_dirs(candidate):
            return candidate

    raise FileNotFoundError(
        "No se encontro el dataset de dientes en formato Supervisely. "
        f"Revisa que {root} contenga carpetas ann/ e img/."
    )


def tooth_class_id(class_title: object) -> int | None:
    title = str(class_title).strip()
    if not title.isdigit():
        return None
    value = int(title)
    if 1 <= value <= 32:
        return value - 1
    return None


def audit_teeth_dataset(root: Path, supervisely_dir: Path) -> dict:
    ann_dir = supervisely_dir / "ann"
    img_dir = supervisely_dir / "img"
    human_dir = supervisely_dir / "masks_human"
    machine_dir = supervisely_dir / "masks_machine"
    annotations = sorted(ann_dir.glob("*.json"))
    images = image_files(img_dir)
    expected_images = {path.name[:-5] for path in annotations if path.name.endswith(".json")}
    actual_images = {path.name for path in images}

    class_counts: Counter[int] = Counter()
    shape_counts: Counter[str] = Counter()
    object_counts = []
    ignored_objects = 0

    for ann_path in annotations:
        data = read_json(ann_path)
        objects = data.get("objects", [])
        object_counts.append(len(objects))
        for obj in objects:
            shape_counts[str(obj.get("geometryType"))] += 1
            class_id = tooth_class_id(obj.get("classTitle"))
            points = obj.get("points", {}).get("exterior", [])
            if obj.get("geometryType") == "polygon" and class_id is not None and len(points) >= 3:
                class_counts[class_id] += 1
            else:
                ignored_objects += 1

    source_info = {
        "images": len(images),
        "annotations": len(annotations),
        "masks_human": len(image_files(human_dir)),
        "masks_machine": len(image_files(machine_dir)),
        "missing_images": len(expected_images - actual_images),
        "extra_images": len(actual_images - expected_images),
        "objects": sum(class_counts.values()),
        "ignored_objects": ignored_objects,
        "objects_per_image_min": min(object_counts) if object_counts else 0,
        "objects_per_image_max": max(object_counts) if object_counts else 0,
        "objects_per_image_avg": round(sum(object_counts) / len(object_counts), 2) if object_counts else 0,
        "shape_counts": dict(shape_counts),
        "sampled_class_counts": {
            f"{class_id}:{TEETH_CLASS_NAMES[class_id]}": count
            for class_id, count in sorted(class_counts.items())
        },
    }

    return {
        "dataset_name": TEETH_DATASET_NAME,
        "root": str(root),
        "selected_yolo_dir": str(supervisely_dir),
        "type": "Supervisely tooth segmentation polygons",
        "classes": TEETH_CLASS_NAMES.copy(),
        "splits": {"source": source_info},
        "notes": [
            "Este dataset trae poligonos por pieza dental y clases numericas 1-32.",
            "Es mejor para entrenar segmentacion/rotulado de dientes que el dataset de hallazgos dentales.",
            "Las clases no son FDI 11-48; representan numeracion 1-32 del propio dataset.",
        ],
    }


def normalized_polygon(points: list, width: int, height: int) -> list[float]:
    values: list[float] = []
    for point in points:
        if len(point) < 2:
            continue
        x = min(max(float(point[0]), 0.0), float(width))
        y = min(max(float(point[1]), 0.0), float(height))
        values.extend([x / width, y / height])
    return values


def write_teeth_label(ann_path: Path, label_path: Path) -> tuple[int, int]:
    data = read_json(ann_path)
    size = data.get("size", {})
    width = int(size.get("width", 0))
    height = int(size.get("height", 0))
    lines = []
    ignored = 0

    if width <= 0 or height <= 0:
        label_path.write_text("", encoding="utf-8")
        return 0, len(data.get("objects", []))

    for obj in data.get("objects", []):
        class_id = tooth_class_id(obj.get("classTitle"))
        points = obj.get("points", {}).get("exterior", [])
        if obj.get("geometryType") != "polygon" or class_id is None or len(points) < 3:
            ignored += 1
            continue
        coords = normalized_polygon(points, width, height)
        if len(coords) < 6:
            ignored += 1
            continue
        coord_text = " ".join(f"{value:.6f}" for value in coords)
        lines.append(f"{class_id} {coord_text}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), ignored


def link_or_copy_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def split_supervisely_annotations(annotations: list[Path]) -> dict[str, list[Path]]:
    shuffled = list(annotations)
    random.Random(SEED).shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * 0.80))
    valid_count = max(1, int(len(shuffled) * 0.10))
    return {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + valid_count],
        "test": shuffled[train_count + valid_count :],
    }


def prepare_teeth_supervisely_yolo(supervisely_dir: Path, output_root: Path) -> tuple[Path, dict]:
    ann_dir = supervisely_dir / "ann"
    img_dir = supervisely_dir / "img"
    output_root.mkdir(parents=True, exist_ok=True)
    annotations = sorted(ann_dir.glob("*.json"))
    splits = split_supervisely_annotations(annotations)
    summary = {
        "prepared_yaml": str(output_root / "data.yaml"),
        "dataset_root": str(output_root),
        "source_root": str(supervisely_dir),
        "splits": {},
    }

    for split, split_annotations in splits.items():
        image_out_dir = output_root / split / "images"
        label_out_dir = output_root / split / "labels"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
        converted_objects = 0
        ignored_objects = 0
        used_images = 0

        for ann_path in split_annotations:
            image_name = ann_path.name[:-5]
            image_path = img_dir / image_name
            if not image_path.exists():
                continue
            link_or_copy_image(image_path, image_out_dir / image_name)
            label_count, ignored_count = write_teeth_label(
                ann_path,
                label_out_dir / f"{Path(image_name).stem}.txt",
            )
            converted_objects += label_count
            ignored_objects += ignored_count
            used_images += 1

        summary["splits"][split] = {
            "images": used_images,
            "labels": len(list(label_out_dir.glob("*.txt"))),
            "objects": converted_objects,
            "ignored_objects": ignored_objects,
        }

    data = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": TEETH_CLASS_NAMES,
    }
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return data_yaml, summary


def audit_segmentation_dataset(root: Path, yolo_dir: Path) -> dict:
    names = load_class_names(yolo_dir)
    splits = {}

    for split in ("train", "valid", "test"):
        img_dir = yolo_dir / split / "images"
        label_dir = yolo_dir / split / "labels"
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        class_counts: Counter[int] = Counter()
        sampled_objects = 0
        sample_labels = []

        for label_path in labels[:2000]:
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                sampled_objects += 1
                class_counts[int(float(parts[0]))] += 1
                if len(sample_labels) < 3:
                    sample_labels.append(line[:240])

        splits[split] = {
            "images": len(image_files(img_dir)),
            "labels": len(labels),
            "sampled_segmentation_objects": sampled_objects,
            "sampled_class_counts": {
                f"{class_id}:{names.get(class_id, class_id)}": count
                for class_id, count in sorted(class_counts.items())
            },
            "sample_labels": sample_labels,
        }

    return {
        "dataset_name": SEGMENTATION_DATASET_NAME,
        "root": str(root),
        "selected_yolo_dir": str(yolo_dir),
        "type": "YOLO segmentation polygons",
        "classes": names,
        "splits": splits,
        "notes": [
            "Este dataset trae segmentacion real en formato YOLO-seg para hallazgos/problemas dentales.",
            "Sirve para demostrar entrenamiento de segmentacion, pero no trae numeracion supervisada por pieza dental.",
        ],
    }


def write_audit_files(audit: dict, out_dir: Path) -> None:
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Analisis del dataset de entrenamiento",
        "",
        f"Dataset local usado: `{audit['dataset_name']}`",
        "",
        f"Formato: `{audit['type']}`",
        "",
        f"Directorio fuente: `{audit['selected_yolo_dir']}`",
        "",
        "## Notas",
        "",
    ]
    for note in audit.get("notes", []):
        lines.append(f"- {note}")

    lines.extend(["", "## Splits"])

    for split, info in audit["splits"].items():
        if "annotations" in info:
            lines.append(
                f"- {split}: {info['images']} imagenes, {info['annotations']} anotaciones, "
                f"{info['objects']} objetos/poligonos"
            )
        else:
            lines.append(f"- {split}: {info['images']} imagenes, {info['labels']} labels")

    lines.extend(
        [
            "",
            "## Clases",
            "",
        ]
    )
    for class_id, name in audit["classes"].items():
        lines.append(f"- {class_id}: {name}")

    (out_dir / "DATASET_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def make_local_yolo_yaml(yolo_dir: Path, output_yaml: Path) -> tuple[Path, dict]:
    names = load_class_names(yolo_dir)
    data = {
        "path": str(yolo_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images" if (yolo_dir / "test" / "images").exists() else "valid/images",
        "names": names,
    }
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary = {
        "prepared_yaml": str(output_yaml),
        "dataset_root": str(yolo_dir),
        "splits": {
            split: {
                "images": len(image_files(yolo_dir / split / "images")),
                "labels": len(list((yolo_dir / split / "labels").glob("*.txt"))),
            }
            for split in ("train", "valid", "test")
        },
    }
    return output_yaml, summary


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
            row["instance_id"] = f"{'U' if jaw == 'upper' else 'L'}{index:02d}"
            ordered_rows.append(row)
    return ordered_rows


def draw_result(result, rows: list[dict], output_path: Path) -> None:
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
    ]
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
            color = palette[row["class_id"] % len(palette)]
            overlay[mask > 0.5] = color

    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    for row in rows:
        x1, y1, x2, y2 = [int(round(row[key])) for key in ("x1", "y1", "x2", "y2")]
        color = palette[row["class_id"] % len(palette)]
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


def predict_and_report(model_path: Path, source_images: list[Path], working_dir: Path, device) -> Path:
    output_dir = working_dir / "predictions_labeled_segmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in source_images],
        imgsz=int(os.getenv("IMGSZ", "512")),
        conf=float(os.getenv("CONF", "0.10")),
        iou=float(os.getenv("IOU", "0.50")),
        device=device,
        save=True,
        project=str(output_dir),
        name="raw_yolo_outputs",
        exist_ok=True,
    )

    report_path = output_dir / "tooth_instances_report.csv"
    fields = ["image", "instance_id", "jaw", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            rows = result_rows(result)
            image_name = Path(result.path).name
            draw_result(result, rows, output_dir / f"{Path(image_name).stem}_segmented_labeled.jpg")
            for row in rows:
                writer.writerow(
                    {
                        "image": image_name,
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
    return output_dir


def select_demo_images(yolo_dir: Path, limit: int) -> list[Path]:
    if os.getenv("SOURCE_IMAGE"):
        return [Path(os.environ["SOURCE_IMAGE"])]
    candidates = image_files(yolo_dir / "test" / "images")
    if not candidates:
        candidates = image_files(yolo_dir / "valid" / "images")
    return candidates[:limit]


def main() -> None:
    seed_everything()
    work_dir = project_dir()
    dataset_kind, dataset_root = choose_training_dataset()

    if dataset_kind == "teeth":
        supervisely_dir = find_teeth_supervisely_dir(dataset_root)
        audit = audit_teeth_dataset(dataset_root, supervisely_dir)
        write_audit_files(audit, work_dir)
        data_yaml, preparation_summary = prepare_teeth_supervisely_yolo(supervisely_dir, PREPARED_TEETH_DIR)
        yolo_dir = data_yaml.parent
    else:
        yolo_dir = find_segmentation_yolo_dir(dataset_root)
        audit = audit_segmentation_dataset(dataset_root, yolo_dir)
        write_audit_files(audit, work_dir)
        data_yaml, preparation_summary = make_local_yolo_yaml(
            yolo_dir,
            work_dir / "prepared_yolo_seg" / "data.yaml",
        )

    (work_dir / "preparation_summary.json").write_text(
        json.dumps(preparation_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Tipo de dataset:", dataset_kind)
    print("Dataset usado:", dataset_root)
    print("YOLO segmentado/preparado:", yolo_dir)
    print("YAML local:", data_yaml)
    print("Preparacion:", json.dumps(preparation_summary, indent=2, ensure_ascii=False))

    if os.getenv("ANALYZE_ONLY", "0") == "1":
        print("ANALYZE_ONLY=1: se detiene despues de auditar y preparar el dataset.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(os.getenv("BASE_MODEL", "yolov8n-seg.pt"))
    default_run_name = "tooth_piece_classifier" if dataset_kind == "teeth" else "treatment_detector"
    run_name = os.getenv("RUN_NAME", default_run_name)
    runs_dir = PROJECT_ROOT / "models"
    model.train(
        data=str(data_yaml),
        epochs=int(os.getenv("EPOCHS", "15")),
        imgsz=int(os.getenv("IMGSZ", "512")),
        batch=int(os.getenv("BATCH", "16")),
        workers=int(os.getenv("WORKERS", "2")),
        patience=int(os.getenv("PATIENCE", "6")),
        device=device,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        seed=SEED,
        plots=True,
        amp=torch.cuda.is_available(),
    )

    best_model = runs_dir / run_name / "weights" / "best.pt"
    demo_images = select_demo_images(yolo_dir, int(os.getenv("PREDICT_LIMIT", "24")))
    if best_model.exists() and demo_images:
        pred_dir = predict_and_report(best_model, demo_images, work_dir, device)
        print("Predicciones rotuladas/segmentadas:", pred_dir)

    zip_path = shutil.make_archive(str(work_dir / "dental_xray_project_outputs"), "zip", root_dir=work_dir)
    print("ZIP final:", zip_path)


if __name__ == "__main__":
    main()
