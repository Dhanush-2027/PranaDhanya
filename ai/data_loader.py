import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from ai.utils import logger
from ai.config import DATASET_DIR, RANDOM_SEED, IMAGE_HEIGHT, IMAGE_WIDTH, DEFAULT_BATCH_SIZE
from ai.preprocessing import load_and_preprocess_image, remove_corrupted_images

def detect_datasets(base_dir=DATASET_DIR):
    """
    Scans base_dir recursively to identify tabular and image datasets.
    Returns:
        tabular_datasets: dict mapping dataset_name -> list of CSV file paths
        image_datasets: dict mapping dataset_name -> data directory path
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.error(f"Datasets directory not found: {base_path}")
        return {}, {}
        
    tabular_datasets = {}
    image_datasets = {}
    
    # 1. Scan for tabular datasets (CSV files)
    csv_files = list(base_path.rglob("*.csv"))
    for csv_path in csv_files:
        # Ignore checkpoints, hidden folders, temp files
        if ".ipynb_checkpoints" in str(csv_path) or csv_path.name.startswith("._"):
            continue
        # Use parent folder name as dataset name
        dataset_name = csv_path.parent.name
        # If there are multiple csvs, we can distinguish them or append
        if dataset_name not in tabular_datasets:
            tabular_datasets[dataset_name] = []
        tabular_datasets[dataset_name].append(csv_path)
        
    # 2. Scan for image datasets
    # An image dataset is a folder containing subfolders (classes) which contain images
    # We look for directories that have subdirectories, and check if those subdirectories contain image files
    for path in base_path.glob("**/"):
        if path == base_path or ".venv" in str(path) or ".git" in str(path) or "models" in str(path):
            continue
        # Check if this folder has subdirectories
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if not subdirs:
            continue
            
        # Check if the subdirectories contain image files
        has_images = False
        for subdir in subdirs:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
            for ext in image_extensions:
                if list(subdir.glob(ext)):
                    has_images = True
                    break
            if has_images:
                break
                
        if has_images:
            # Avoid duplicate image paths or parent paths already registered
            dataset_name = path.name
            # If it's train/val/test split folder, record parent as dataset
            if dataset_name in ["train", "val", "test", "validation", "data"]:
                dataset_name = path.parent.name
                image_datasets[dataset_name] = path.parent
            else:
                image_datasets[dataset_name] = path
                
    logger.info(f"Detected tabular datasets: {list(tabular_datasets.keys())}")
    logger.info(f"Detected image datasets: {list(image_datasets.keys())}")
    return tabular_datasets, image_datasets

def detect_target_column(df, dataset_name):
    """Automatically detect the target column for tabular datasets."""
    cols_lower = [c.lower() for c in df.columns]
    
    # 1. Match typical target names for crop / fertilizer / yield / price
    if 'crop' in dataset_name.lower():
        if 'label' in df.columns: return 'label'
        if 'crop' in df.columns: return 'crop'
    if 'fertilizer' in dataset_name.lower():
        for c in df.columns:
            if 'fertilizer' in c.lower(): return c
    if 'yield' in dataset_name.lower():
        for c in df.columns:
            if 'yield' in c.lower(): return c
    if 'price' in dataset_name.lower():
        for c in df.columns:
            if 'price' in c.lower(): return c
            
    # 2. General target names
    for name in ['target', 'label', 'class', 'output', 'y', 'fertilizer_name']:
        for c in df.columns:
            if c.lower() == name:
                return c
                
    # 3. Fallback: last column
    return df.columns[-1]

def split_tabular_data(df, target_col, is_classification=True):
    """
    Splits tabular DataFrame into train (80%), val (10%), and test (10%) splits.
    Applies stratified split for classification if possible.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if len(df) < 10:
        # Graceful fallback for extremely small datasets (e.g. dummy test files)
        logger.warning(f"Dataset has only {len(df)} rows. Skipping splitting to avoid ValueError.")
        return X, y, X, y, X, y
        
    # Determine if we can do stratified split
    stratify = y if is_classification else None
    if is_classification:
        # Verify if all classes have at least 2 samples
        class_counts = y.value_counts()
        if (class_counts < 2).any():
            logger.warning("Some classes have fewer than 2 samples. Disabling stratification.")
            stratify = None
            
    # Split: Train (80%) vs Temp (20%)
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
        )
    except Exception as e:
        logger.warning(f"Stratified split failed: {e}. Defaulting to non-stratified split.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=None
        )
        
    # Split Temp: Val (10% of total) and Test (10% of total) -> 50% of Temp
    temp_stratify = y_temp if (is_classification and stratify is not None) else None
    if temp_stratify is not None:
        temp_class_counts = y_temp.value_counts()
        if (temp_class_counts < 2).any():
            temp_stratify = None
            
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_stratify
        )
    except Exception as e:
        logger.warning(f"Temp stratified split failed: {e}. Defaulting to non-stratified.")
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=None
        )
        
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_image_paths_and_labels(data_dir):
    """
    Recursively scans data_dir for image files and parses class labels based on subdirectories.
    Returns:
        image_paths: list of strings (absolute paths)
        labels: list of ints (class indices)
        class_names: list of strings (ordered class labels)
    """
    data_path = Path(data_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    
    # 1. Clean corrupted images first
    remove_corrupted_images(data_path)
    
    # Find all class subdirectories
    # We support flat structures (data_dir/class_name/img.jpg) or nested train folders (data_dir/train/class_name/img.jpg)
    class_dirs = sorted([d for d in data_path.glob("**/") if d.is_dir() and d != data_path])
    
    # Filter class dirs to only those containing images directly
    valid_class_dirs = []
    for d in class_dirs:
        # Ignore train/val/test wrapper directories
        if d.name in ["train", "val", "test", "validation", "data"]:
            continue
        has_img = False
        for ext in image_extensions:
            if list(d.glob(ext)):
                has_img = True
                break
        if has_img:
            valid_class_dirs.append(d)
            
    # Unique class names
    class_names = sorted(list(set([d.name for d in valid_class_dirs])))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    for d in valid_class_dirs:
        class_idx = class_to_idx[d.name]
        for ext in image_extensions:
            for img_path in d.glob(ext):
                image_paths.append(str(img_path.resolve()))
                labels.append(class_idx)
                
    logger.info(f"Loaded image paths. Found {len(image_paths)} images across {len(class_names)} classes.")
    return image_paths, labels, class_names

def build_image_datasets(image_paths, labels, class_names, batch_size=DEFAULT_BATCH_SIZE):
    """
    Splits image paths/labels into Train (80%), Val (10%), and Test (10%),
    and creates TF datasets optimized for performance.
    """
    if len(image_paths) < 10:
        logger.warning(f"Image dataset has only {len(image_paths)} files. Skipping splitting to avoid ValueError.")
        train_paths, val_paths, test_paths = image_paths, image_paths, image_paths
        train_labels, val_labels, test_labels = labels, labels, labels
    else:
        # Stratified split for images
        try:
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                image_paths, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
            )
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_labels
            )
        except Exception as e:
            logger.warning(f"Stratified split failed for images: {e}. Performing default split.")
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                image_paths, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=None
            )
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, test_size=0.5, random_state=RANDOM_SEED, stratify=None
            )
        
    logger.info(f"Image Splits: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    
    # Create TensorFlow datasets
    def make_dataset(paths, lbls):
        ds = tf.data.Dataset.from_tensor_slices((paths, lbls))
        ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        return ds
        
    train_ds = make_dataset(train_paths, train_labels)
    val_ds = make_dataset(val_paths, val_labels)
    test_ds = make_dataset(test_paths, test_labels)
    
    # Configure performance optimization
    train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, val_labels, test_labels
