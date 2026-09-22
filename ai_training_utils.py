import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

def scan_tabular_dataset(file_path, target_col, is_classification=True):
    """
    Scans a tabular dataset and returns a summary report.
    """
    print(f"Scanning tabular dataset: {file_path}", flush=True)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    total_samples = len(df)
    missing_values = int(df.isnull().sum().sum())
    duplicate_records = int(df.duplicated().sum())
    corrupted_records = 0 # No corrupted images for tabular
    
    # Check for unreadable rows/records
    valid_rows = df.dropna()
    corrupted_records = total_samples - len(valid_rows)
    
    summary = {
        "total_samples": total_samples,
        "missing_values": missing_values,
        "duplicate_records": duplicate_records,
        "corrupted_records": corrupted_records,
    }
    
    if is_classification:
        classes = sorted(df[target_col].unique().tolist())
        samples_per_class = df[target_col].value_counts().to_dict()
        samples_per_class = {str(k): int(v) for k, v in samples_per_class.items()}
        
        min_class = min(samples_per_class, key=samples_per_class.get)
        max_class = max(samples_per_class, key=samples_per_class.get)
        min_count = samples_per_class[min_class]
        max_count = samples_per_class[max_class]
        
        summary.update({
            "classes": classes,
            "samples_per_class": samples_per_class,
            "class_imbalance_statistics": {
                "min_class": min_class,
                "min_count": min_count,
                "max_class": max_class,
                "max_count": max_count,
                "ratio_min_max": float(min_count / max_count) if max_count > 0 else 0.0
            }
        })
    else:
        summary.update({
            "classes": [],
            "samples_per_class": {},
            "class_imbalance_statistics": {}
        })
        
    return df, summary

def _verify_and_hash_image(f_path):
    """Helper function to run in thread pool."""
    try:
        with Image.open(f_path) as img:
            img.verify()
        with open(f_path, 'rb') as fh:
            f_hash = hashlib.sha256(fh.read()).hexdigest()
        return f_path, f_hash, False
    except Exception:
        return f_path, None, True

def scan_image_dataset(dataset_dir):
    """
    Scans an image classification dataset directory in parallel.
    """
    print(f"Scanning image dataset: {dataset_dir}", flush=True)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    samples_per_class = {}
    total_samples = 0
    corrupted_records = 0
    duplicate_records = 0
    seen_hashes = set()
    
    # Collect all image files
    all_file_tasks = []
    file_to_class = {}
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        samples_per_class[cls] = len(files)
        total_samples += len(files)
        
        for f in files:
            f_path = os.path.join(cls_dir, f)
            all_file_tasks.append(f_path)
            file_to_class[f_path] = cls
            
    print(f"Submitting {len(all_file_tasks)} files to thread pool for verification...", flush=True)
    
    # Use ThreadPoolExecutor for concurrent IO and hashing
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(_verify_and_hash_image, path): path for path in all_file_tasks}
        
        for count, future in enumerate(as_completed(futures)):
            path, f_hash, is_corrupted = future.result()
            
            if is_corrupted:
                corrupted_records += 1
                # Deduct from samples per class
                cls = file_to_class[path]
                samples_per_class[cls] -= 1
            else:
                if f_hash in seen_hashes:
                    duplicate_records += 1
                    # Deduct duplicate from samples per class to represent unique valid
                    cls = file_to_class[path]
                    samples_per_class[cls] -= 1
                else:
                    seen_hashes.add(f_hash)
                    
            if (count + 1) % 20000 == 0:
                print(f"Scanned {count + 1}/{len(all_file_tasks)} images...", flush=True)
                
    min_class = min(samples_per_class, key=samples_per_class.get) if classes else ""
    max_class = max(samples_per_class, key=samples_per_class.get) if classes else ""
    min_count = samples_per_class[min_class] if classes else 0
    max_count = samples_per_class[max_class] if classes else 0
    
    summary = {
        "classes": classes,
        "samples_per_class": samples_per_class,
        "total_samples": total_samples,
        "missing_values": 0,
        "duplicate_records": duplicate_records,
        "corrupted_records": corrupted_records,
        "class_imbalance_statistics": {
            "min_class": min_class,
            "min_count": min_count,
            "max_class": max_class,
            "max_count": max_count,
            "ratio_min_max": float(min_count / max_count) if max_count > 0 else 0.0
        }
    }
    return summary

def validate_tabular_data(df, expected_features, target_col):
    """
    Validates features and label consistency before training.
    """
    print("Validating tabular data...", flush=True)
    # Verify target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    # Verify features exist
    for col in expected_features:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found.")
            
    # Verify feature types are numeric or easily encodable
    for col in expected_features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # If object/categorical, make sure it is not all missing
            if df[col].isnull().all():
                raise ValueError(f"Feature column '{col}' contains all missing values.")
                
    # Verify label consistency (no all nulls)
    if df[target_col].isnull().all():
        raise ValueError(f"Target column '{target_col}' contains all missing values.")
        
    print("Tabular data validation PASSED.", flush=True)
