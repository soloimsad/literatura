from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOOTH_CLASSES = (
    "11 numara",
    "12 numara",
    "13 numara",
    "14 numara",
    "15 numara",
    "16 numara",
    "17 numara",
    "18 numara",
    "21 numara",
    "22 numara",
    "23 numara",
    "24 numara",
    "25 numara",
    "26 numara",
    "27 numara",
    "28 numara",
    "31 numara",
    "32 numara",
    "33 numara",
    "34 numara",
    "35 numara",
    "36 numara",
    "37 numara",
    "38 numara",
    "41 numara",
    "42 numara",
    "43 numara",
    "44 numara",
    "45 numara",
    "46 numara",
    "47 numara",
    "48 numara",
    "51 numara",
    "52 numara",
    "53 numara",
    "54 numara",
    "55 numara",
    "61 numara",
    "62 numara",
    "63 numara",
    "64 numara",
    "65 numara",
    "71 numara",
    "72 numara",
    "73 numara",
    "74 numara",
    "75 numara",
    "81 numara",
    "82 numara",
    "83 numara",
    "84 numara",
    "85 numara",
    "Paramolar",
    "Unidentified tooth",
)

TREATMENT_CLASSES = (
    "Caries",
    "Crown",
    "Filling",
    "Implant",
    "Malaligned",
    "Mandibular Canal",
    "Missing teeth",
    "Periapical lesion",
    "Retained root",
    "Root Canal Treatment",
    "Root Piece",
    "Impacted tooth",
    "Maxillary sinus",
    "Bone Loss",
    "Fracture teeth",
    "Permanent Teeth",
    "Supra Eruption",
    "TAD",
    "Abutment",
    "Attrition",
    "Bone defect",
    "Gingival former",
    "Metal band",
    "Orthodontic brackets",
    "Permanent retainer",
    "Post-core",
    "Plating",
    "Wire",
    "Cyst",
    "Root resorption",
    "Primary teeth",
)


@dataclass(frozen=True)
class ModelSpec:
    role: str
    display_name: str
    weights: Path
    output_dir: str
    description: str
    expected_classes: tuple[str, ...]

    @property
    def exists(self) -> bool:
        return self.weights.exists()


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "tooth": ModelSpec(
        role="tooth",
        display_name="Clasificacion de pieza dental",
        weights=PROJECT_ROOT / "models" / "tooth_piece_classifier" / "weights" / "best.pt",
        output_dir="tooth_piece_classifier",
        description="Modelo YOLO-seg para segmentar piezas dentales y clasificar numero FDI.",
        expected_classes=TOOTH_CLASSES,
    ),
    "treatment": ModelSpec(
        role="treatment",
        display_name="Posible tratamiento o hallazgo",
        weights=PROJECT_ROOT / "models" / "treatment_detector" / "weights" / "best.pt",
        output_dir="treatment_detector",
        description="Modelo YOLO para detectar hallazgos dentales asociados a posibles tratamientos.",
        expected_classes=TREATMENT_CLASSES,
    ),
}


def get_model_spec(role: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[role]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Rol de modelo invalido: {role}. Usa uno de: {valid}.") from exc


def registry_rows() -> list[dict[str, str]]:
    rows = []
    for spec in MODEL_REGISTRY.values():
        rows.append(
            {
                "role": spec.role,
                "display_name": spec.display_name,
                "weights": str(spec.weights),
                "available": "yes" if spec.exists else "no",
                "description": spec.description,
            }
        )
    return rows
