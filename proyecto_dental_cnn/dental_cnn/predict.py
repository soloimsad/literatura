"""Segment and label teeth/dental findings in a panoramic radiograph."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import load_grayscale
from .model import build_model
from .postprocess import (
    extract_instances_from_binary_mask,
    extract_instances_from_class_map,
)
from .visualize import draw_instances, save_instances_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input dental radiography image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt.")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "segmented.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--label-style", choices=["sequential", "fdi"], default="sequential")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    mask_mode = checkpoint["mask_mode"]
    image_size = tuple(checkpoint["image_size"])
    base_channels = int(checkpoint.get("base_channels", 16))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mask_mode, len(class_names), base_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    original = load_grayscale(args.image)
    model_input = preprocess(original, image_size).to(device)

    with torch.no_grad():
        logits = model(model_input)[0].cpu()

    if mask_mode == "binary":
        prob_small = torch.sigmoid(logits[0]).numpy()
        prob = cv2.resize(prob_small, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = (prob >= args.threshold).astype(np.uint8)
        instances = extract_instances_from_binary_mask(
            mask,
            prob,
            min_area=args.min_area,
            label_style=args.label_style,
        )
    else:
        probs_small = torch.softmax(logits, dim=0).numpy()
        class_map_small = np.argmax(probs_small, axis=0).astype(np.uint8)
        class_map = cv2.resize(
            class_map_small,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        probs = np.stack(
            [
                cv2.resize(channel, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
                for channel in probs_small
            ],
            axis=0,
        )
        confident = np.max(probs, axis=0) >= args.threshold
        class_map = np.where(confident, class_map, 0).astype(np.uint8)
        instances = extract_instances_from_class_map(
            class_map,
            class_names,
            probabilities=probs,
            min_area=args.min_area,
            label_style=args.label_style,
        )

    draw_instances(args.image, instances, args.output)
    save_instances_json(instances, args.output.with_suffix(".json"))
    print(f"Saved image: {args.output}")
    print(f"Saved detections: {args.output.with_suffix('.json')}")
    print(f"Instances: {len(instances)}")


def preprocess(image: np.ndarray, image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - 0.5) / 0.5
    tensor = torch.from_numpy(resized[None, None, :, :])
    return tensor.float()


if __name__ == "__main__":
    main()
