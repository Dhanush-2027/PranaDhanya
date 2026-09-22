import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ai.utils import logger
from ai.config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, RANDOM_SEED

def preprocess_tabular_dataset(df, target_col, output_dir, is_classification=True):
    """
    Cleans, preprocesses and standardizes tabular datasets.
    Handles duplicate removal, missing value imputation, incorrect type detection,
    outlier removal, label encoding, numerical standardization, and class balancing.
    """
    logger.info(f"Preprocessing tabular dataset. Original shape: {df.shape}")
    
    # 1. Remove duplicate records
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Shape after removing duplicates: {df.shape}")
    
    # 2. Drop rows where the target is missing
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    # 3. Detect incorrect data types & handle missing values / corrupted rows
    # Convert numerical-looking columns to numeric
    feature_cols = [c for c in df.columns if c != target_col]
    
    for col in feature_cols:
        # Check if the column is mostly numbers (excluding categorical text)
        non_null_vals = df[col].dropna()
        if len(non_null_vals) > 0:
            sample_val = non_null_vals.iloc[0]
            if isinstance(sample_val, (int, float)) or (isinstance(sample_val, str) and sample_val.replace('.', '', 1).isdigit()):
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Separate numeric and categorical features
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    logger.info(f"Detected numeric features: {numeric_cols}")
    logger.info(f"Detected categorical features: {categorical_cols}")
    
    # Fill missing values
    for col in numeric_cols:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
        
    for col in categorical_cols:
        mode_series = df[col].mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
        df[col] = df[col].fillna(mode_val)
        
    # Remove corrupted rows (rows that still contain NaNs in features)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Shape after handling missing/corrupted rows: {df.shape}")
    
    # 4. Outlier removal using IQR method for numerical columns
    if len(numeric_cols) > 0 and len(df) > 50:
        before_outliers = len(df)
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            # Keep values within range
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        df = df.reset_index(drop=True)
        logger.info(f"Outlier removal (IQR) filtered {before_outliers - len(df)} records. New shape: {df.shape}")
        
    # 5. Convert categorical columns to numerical form
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        # Save label classes as list for JSON compatibility
        encoder_mapping_path = output_dir / f"label_encoder_{col}.json"
        with open(encoder_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(list(le.classes_), f)
            
    # Encode target column if classification
    target_encoder = None
    if is_classification:
        target_encoder = LabelEncoder()
        df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))
        # Save label classes for decoding predictions
        classes_path = output_dir / "label_classes.json"
        with open(classes_path, 'w', encoding='utf-8') as f:
            json.dump(list(target_encoder.classes_), f)
        logger.info(f"Target variable classes: {list(target_encoder.classes_)}")
    else:
        # Regression target
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        
    # 6. Normalize/Standardize numerical features
    scaler = None
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        # Save scaler to output_dir
        joblib.dump(scaler, output_dir / "scaler.pkl")
        logger.info("Saved scaler.pkl for features")
        
    # Save the order of feature columns so prediction pipeline matches it
    feature_names = [c for c in df.columns if c != target_col]
    with open(output_dir / "feature_columns.json", 'w', encoding='utf-8') as f:
        json.dump(feature_names, f)
        
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    return df, label_encoders, target_encoder, scaler, feature_names

def balance_tabular_classes(X, y):
    """
    Balances class distributions.
    Tries Random Oversampling if imbalance is detected.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    logger.info(f"Class distribution before balancing: {class_counts}")
    
    max_count = max(counts)
    min_count = min(counts)
    
    # If the imbalance ratio is high (> 1.5), apply Random Oversampling
    if max_count / max(1, min_count) > 1.5:
        logger.info("Imbalance detected. Performing Random Oversampling...")
        X_resampled = []
        y_resampled = []
        
        for cls in unique:
            cls_indices = np.where(y == cls)[0]
            # Oversample to match max_count
            oversampled_indices = np.random.choice(cls_indices, size=max_count, replace=True)
            
            # If X is pandas DataFrame or numpy array
            if isinstance(X, pd.DataFrame):
                X_resampled.append(X.iloc[oversampled_indices])
            else:
                X_resampled.append(X[oversampled_indices])
            y_resampled.append(y[oversampled_indices] if isinstance(y, np.ndarray) else y.iloc[oversampled_indices])
            
        if isinstance(X, pd.DataFrame):
            X = pd.concat(X_resampled).reset_index(drop=True)
        else:
            X = np.concatenate(X_resampled)
            
        if isinstance(y, np.ndarray):
            y = np.concatenate(y_resampled)
        else:
            y = pd.concat(y_resampled).reset_index(drop=True)
            
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution after balancing: {dict(zip(unique, counts))}")
        
    return X, y

def remove_corrupted_images(image_dir):
    """
    Scans an image directory, attempts to load every image using PIL,
    and removes any corrupted or unreadable images.
    """
    image_dir = Path(image_dir)
    corrupted_count = 0
    total_count = 0
    
    for img_path in list(image_dir.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            total_count += 1
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify image health
                # Attempt to load the image data
                with Image.open(img_path) as img:
                    img.load()
            except Exception as e:
                logger.warning(f"Corrupted image found and removed: {img_path}. Error: {e}")
                try:
                    os.remove(img_path)
                except Exception:
                    pass
                corrupted_count += 1
                
    logger.info(f"Verified {total_count} images. Removed {corrupted_count} corrupted images.")

def load_and_preprocess_image(image_path, label, channels=IMAGE_CHANNELS):
    """
    TensorFlow mapping function to load and preprocess a single image path.
    Resizes image and normalizes pixel values to [0, 1].
    """
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_raw, channels=channels)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = image / 255.0  # Normalize to [0, 1]
    return image, label
