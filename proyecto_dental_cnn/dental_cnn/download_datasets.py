"""Download the Kaggle datasets used by the project."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASET_SLUGS = (
    "imtkaggleteam/dental-radiography",
    "lokisilvres/dental-disease-panoramic-detection-dataset",
)


def download_dataset(slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Run: python -m pip install kagglehub"
        ) from exc

    path = Path(kagglehub.dataset_download(slug))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug",
        action="append",
        default=[],
        help="Kaggle dataset slug. Can be repeated. Defaults to the two project datasets.",
    )
    args = parser.parse_args()

    slugs = args.slug or list(DATASET_SLUGS)
    for slug in slugs:
        path = download_dataset(slug)
        print(f"{slug}: {path}")


if __name__ == "__main__":
    main()
