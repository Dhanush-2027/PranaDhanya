#!/usr/bin/env python3
import argparse
import os
import sys
import zipfile
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("ERROR: Kaggle package not installed. Run: pip install kaggle")
    sys.exit(1)

DATASETS = {
    "plant_disease": {
        "slug": "nirmalsankalana/plant-diseases-training-dataset",
        "folder": "datasets/plant_disease",
        "description": "Plant disease images and CSV metadata",
    },
    "crop_recommendation": {
        "slug": "atharvaingle/crop-recommendation-dataset",
        "folder": "datasets/crop_recommendation",
        "description": "Crop recommendation tabular dataset",
    },
    "fertilizer_prediction": {
        "slug": "gdabhishek/fertilizer-prediction",
        "folder": "datasets/fertilizer_prediction",
        "description": "Fertilizer recommendation tabular dataset",
    },
    "cattle_diseases": {
        "slug": "devang03mgr/cattle-diseases-datasets",
        "folder": "datasets/cattle_diseases",
        "description": "Cattle disease images and metadata",
    },
    "dog_skin_disease": {
        "slug": "youssefmohmmed/dogs-skin-diseases-image-dataset",
        "folder": "datasets/dog_skin_disease",
        "description": "Dog skin disease images and metadata",
    },
    "livestock": {
        "slug": "kartikeybartwal/dataset",
        "folder": "datasets/livestock",
        "description": "General livestock and agriculture tabular dataset",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and extract selected Kaggle datasets into the project datasets folder"
    )
    parser.add_argument(
        "--username",
        help="Kaggle username. If not provided, environment variable KAGGLE_USERNAME is used.",
    )
    parser.add_argument(
        "--key",
        help="Kaggle API key. If not provided, environment variable KAGGLE_KEY is used.",
    )
    parser.add_argument(
        "--dest",
        default="datasets",
        help="Root destination path for downloaded datasets. Defaults to datasets/",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        help="Which datasets to download. Defaults to all.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing dataset directory before downloading.",
    )
    return parser.parse_args()


def init_kaggle_api(username: str | None, key: str | None) -> KaggleApi:
    if username:
        os.environ["KAGGLE_USERNAME"] = username
    if key:
        os.environ["KAGGLE_KEY"] = key

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError(
            "Kaggle credentials are required. Set KAGGLE_USERNAME and KAGGLE_KEY in the environment or pass --username and --key."
        )

    api = KaggleApi()
    api.authenticate()
    return api


def clean_directory(path: Path) -> None:
    if not path.exists():
        return

    for child in path.iterdir():
        if child.is_dir():
            clean_directory(child)
            child.rmdir()
        else:
            child.unlink()


def unzip_all_files(destination: Path) -> None:
    for archive in destination.glob("*.zip"):
        print(f"Extracting {archive.name}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(destination)
        archive.unlink()


def download_dataset(api: KaggleApi, dataset_key: str, dest_root: Path, clean: bool) -> None:
    config = DATASETS[dataset_key]
    dest_dir = dest_root / config["folder"].replace("datasets/", "")
    print(f"\nDownloading dataset: {dataset_key}\n  slug: {config['slug']}\n  destination: {dest_dir}")

    if clean and dest_dir.exists():
        print(f"Cleaning existing directory: {dest_dir}")
        clean_directory(dest_dir)
        dest_dir.rmdir()

    dest_dir.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files(
        config["slug"],
        path=str(dest_dir),
        unzip=False,
        quiet=False,
    )

    unzip_all_files(dest_dir)
    print(f"Finished downloading and extracting {dataset_key}.")


def main():
    args = parse_args()
    dest_root = Path(args.dest).resolve()
    selected = [name.strip().lower() for name in args.datasets]

    missing = [name for name in selected if name not in DATASETS]
    if missing:
        print(f"ERROR: Unknown dataset names: {missing}")
        print("Available names:", ", ".join(DATASETS.keys()))
        sys.exit(1)

    api = init_kaggle_api(args.username, args.key)

    for dataset_name in selected:
        download_dataset(api, dataset_name, dest_root, args.clean)

    print("\nAll requested datasets have been downloaded and extracted.")


if __name__ == "__main__":
    main()
