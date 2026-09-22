import os
import pandas as pd
from pathlib import Path

datasets_dir = Path("datasets")
print("=== Tabular Datasets ===")
for folder in ["crop_recommendation", "fertilizer_prediction", "yield_prediction", "price_prediction"]:
    path = datasets_dir / folder
    if path.exists():
        csvs = list(path.glob("*.csv"))
        print(f"\nFolder: {folder}")
        for csv in csvs:
            try:
                df = pd.read_csv(csv)
                print(f"  - {csv.name}: shape={df.shape}")
                print(f"    columns={list(df.columns)}")
                if not df.empty:
                    print(f"    first row={df.iloc[0].to_dict()}")
            except Exception as e:
                print(f"  - {csv.name}: error reading {e}")

print("\n=== Image Datasets ===")
for folder in ["plant_disease", "cattle_diseases", "dog_skin_disease", "livestock"]:
    path = datasets_dir / folder
    if path.exists():
        # count files with image extensions
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
        file_count = 0
        subdirs = []
        for ext in exts:
            file_count += len(list(path.rglob(ext)))
        for item in path.rglob("*"):
            if item.is_dir():
                subdirs.append(item)
        print(f"Folder: {folder} -> {file_count} images, {len(subdirs)} subdirectories")
