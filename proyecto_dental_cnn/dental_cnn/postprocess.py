"""Post-processing utilities for predicted dental masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Instance:
    label: str
    class_name: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    score: float


def clean_binary_mask(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for component_id in range(1, num):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == component_id] = 1
    return cleaned


def extract_instances_from_class_map(
    class_map: np.ndarray,
    class_names: list[str],
    probabilities: np.ndarray | None = None,
    min_area: int = 120,
    label_style: str = "sequential",
) -> list[Instance]:
    instances: list[Instance] = []
    for class_id, class_name in enumerate(class_names, start=1):
        mask = clean_binary_mask((class_map == class_id).astype(np.uint8), min_area=min_area)
        instances.extend(
            _components_to_instances(mask, class_name, probabilities, class_id, min_area)
        )
    return assign_tooth_labels(instances, label_style)


def extract_instances_from_binary_mask(
    mask: np.ndarray,
    probabilities: np.ndarray | None = None,
    min_area: int = 120,
    label_style: str = "sequential",
) -> list[Instance]:
    clean = clean_binary_mask(mask, min_area=min_area)
    instances = _components_to_instances(clean, "Tooth", probabilities, 1, min_area)
    return assign_tooth_labels(instances, label_style)


def _components_to_instances(
    mask: np.ndarray,
    class_name: str,
    probabilities: np.ndarray | None,
    class_id: int,
    min_area: int,
) -> list[Instance]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    instances: list[Instance] = []
    for component_id in range(1, num):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component_mask = (labels == component_id).astype(np.uint8)
        if probabilities is not None and probabilities.ndim == 3 and class_id < probabilities.shape[0]:
            score = float(probabilities[class_id][component_mask > 0].mean())
        elif probabilities is not None and probabilities.ndim == 2:
            score = float(probabilities[component_mask > 0].mean())
        else:
            score = 1.0
        instances.append(
            Instance(
                label="",
                class_name=class_name,
                mask=component_mask,
                bbox=(x, y, x + w, y + h),
                area=area,
                score=score,
            )
        )
    return instances


def assign_tooth_labels(instances: list[Instance], label_style: str = "sequential") -> list[Instance]:
    tooth_like = []
    others = []
    for inst in instances:
        name = inst.class_name.lower()
        if "tooth" in name or "teeth" in name or name == "tooth" or "permanent" in name or "primary" in name:
            tooth_like.append(inst)
        else:
            others.append(inst)

    if tooth_like:
        centers_y = np.array([(inst.bbox[1] + inst.bbox[3]) / 2 for inst in tooth_like], dtype=np.float32)
        jaw_cut = float(np.median(centers_y))
        upper = [inst for inst in tooth_like if (inst.bbox[1] + inst.bbox[3]) / 2 <= jaw_cut]
        upper_ids = {id(inst) for inst in upper}
        lower = [inst for inst in tooth_like if id(inst) not in upper_ids]
        upper.sort(key=lambda inst: (inst.bbox[0] + inst.bbox[2]) / 2)
        lower.sort(key=lambda inst: (inst.bbox[0] + inst.bbox[2]) / 2)

        if label_style == "fdi":
            _assign_fdi(upper, lower)
        else:
            for idx, inst in enumerate(upper + lower, start=1):
                inst.label = f"T{idx:02d}"

    for idx, inst in enumerate(sorted(others, key=lambda item: (item.bbox[1], item.bbox[0])), start=1):
        inst.label = f"{short_label(inst.class_name)}{idx:02d}"
    return sorted(tooth_like + others, key=lambda inst: (inst.bbox[1], inst.bbox[0]))


def _assign_fdi(upper: list[Instance], lower: list[Instance]) -> None:
    upper_codes = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    lower_codes = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
    for inst, code in zip(upper, _fit_codes(upper_codes, len(upper))):
        inst.label = str(code)
    for inst, code in zip(lower, _fit_codes(lower_codes, len(lower))):
        inst.label = str(code)


def _fit_codes(codes: list[int], count: int) -> list[int]:
    if count <= len(codes):
        start = max(0, (len(codes) - count) // 2)
        return codes[start : start + count]
    return codes + list(range(1, count - len(codes) + 1))


def short_label(name: str) -> str:
    words = [word for word in name.replace("-", " ").split() if word]
    if not words:
        return "C"
    return "".join(word[0].upper() for word in words[:3])
