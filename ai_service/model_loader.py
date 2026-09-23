import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path):
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = REPO_ROOT / candidate
    return repo_candidate


class KerasTabularModelWrapper:
    """
    Wraps a Keras tabular model to mimic the predict/predict_proba interface
    of scikit-learn/XGBoost, handling features alignment, scaling, and categorical encoding.
    """
    def __init__(self, h5_path, metadata_dir, is_regression=False):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(h5_path)
        self.metadata_dir = Path(metadata_dir)
        self.is_regression = is_regression
        
        # Load scaler
        scaler_path = self.metadata_dir / 'scaler.pkl'
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        # Load feature columns
        feature_path = self.metadata_dir / 'feature_columns.json'
        if feature_path.exists():
            with open(feature_path, 'r', encoding='utf-8') as f:
                self.feature_columns = json.load(f)
        else:
            self.feature_columns = None
            
        # Load label classes
        classes_path = self.metadata_dir / 'label_classes.json'
        if classes_path.exists():
            with open(classes_path, 'r', encoding='utf-8') as f:
                self.label_classes = json.load(f)
        else:
            self.label_classes = None

    def _preprocess(self, df):
        # Create a copy to prevent mutation warnings
        df_proc = df.copy()
        
        # 1. Align feature columns with case, space, and synonym tolerance
        if self.feature_columns:
            # Map lowercase stripped names to actual input columns
            input_cols_map = {c.lower().strip(): c for c in df_proc.columns}
            
            new_df_data = {}
            for col in self.feature_columns:
                col_key = col.lower().strip()
                if col_key in input_cols_map:
                    new_df_data[col] = df_proc[input_cols_map[col_key]].values
                else:
                    # Agricultural feature synonyms/typo correction
                    synonyms = {
                        'temparature': ['temperature', 'temp'],
                        'temperature': ['temparature', 'temp'],
                        'phosphorous': ['phosphorus', 'phosphorous', 'p'],
                        'phosphorus': ['phosphorous', 'phosphorus', 'p'],
                        'humidity ': ['humidity', 'humid'],
                        'humidity': ['humidity ', 'humid'],
                        'soil type': ['soil_type', 'soiltype', 'soil_type', 'soilType'],
                        'soil_type': ['soil type', 'soiltype', 'soil_type', 'soilType'],
                        'crop type': ['crop_type', 'croptype', 'crop_type', 'cropType'],
                        'crop_type': ['crop type', 'croptype', 'crop_type', 'cropType']
                    }
                    found = False
                    # Look through synonyms for matches
                    for syn in synonyms.get(col_key, []):
                        syn_key = syn.lower().strip()
                        if syn_key in input_cols_map:
                            new_df_data[col] = df_proc[input_cols_map[syn_key]].values
                            found = True
                            break
                    if not found:
                        new_df_data[col] = np.zeros(len(df_proc))
                        
            df_proc = pd.DataFrame(new_df_data)
            
        # 2. Encode categorical columns using saved encoders
        for col in df_proc.columns:
            enc_file = self.metadata_dir / f"label_encoder_{col}.json"
            if enc_file.exists():
                with open(enc_file, 'r', encoding='utf-8') as f:
                    classes = json.load(f)
                val = df_proc[col].iloc[0]
                if isinstance(val, str):
                    val_lower = val.strip().lower().replace(' ', '_').replace('-', '_')
                    # Match case-insensitively
                    idx = 0
                    for c_idx, c_val in enumerate(classes):
                        if c_val.strip().lower().replace(' ', '_').replace('-', '_') == val_lower:
                            idx = c_idx
                            break
                    df_proc[col] = idx
                else:
                    df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0).astype(int)
                    
        # 3. Apply standard scaler to numeric features
        if self.scaler:
            numeric_cols = []
            for col in df_proc.columns:
                enc_file = self.metadata_dir / f"label_encoder_{col}.json"
                if not enc_file.exists():
                    numeric_cols.append(col)
                    
            if len(numeric_cols) == self.scaler.n_features_in_:
                df_proc[numeric_cols] = self.scaler.transform(df_proc[numeric_cols])
            elif df_proc.shape[1] == self.scaler.n_features_in_:
                df_proc.iloc[:, :] = self.scaler.transform(df_proc)
                
        return df_proc.values.astype('float32')

    def predict(self, df):
        X = self._preprocess(df)
        preds = self.model.predict(X, verbose=0)
        if self.is_regression:
            return preds.flatten()
        else:
            idx = np.argmax(preds, axis=1)
            return idx

    def predict_proba(self, df):
        X = self._preprocess(df)
        preds = self.model.predict(X, verbose=0)
        return preds


@lru_cache(maxsize=None)
def load_crop_recommender(path='ai/models/crop_recommendation/crop_recommender.pkl'):
    resolved_path = _resolve_path(path)
    if resolved_path:
        h5_path = resolved_path.with_suffix('.h5')
        if h5_path.exists():
            return KerasTabularModelWrapper(h5_path, resolved_path.parent, is_regression=False)
        if resolved_path.exists():
            return joblib.load(resolved_path)
    return None


@lru_cache(maxsize=None)
def load_crop_metadata(dir_path='ai/models/crop_recommendation'):
    metadata = {}
    resolved_dir = _resolve_path(dir_path)
    if resolved_dir is None:
        return metadata
    feature_path = resolved_dir / 'feature_columns.json'
    label_path = resolved_dir / 'label_classes.json'
    if feature_path.exists():
        with open(feature_path, 'r', encoding='utf-8') as f:
            metadata['feature_columns'] = json.load(f)
    if label_path.exists():
        with open(label_path, 'r', encoding='utf-8') as f:
            metadata['label_classes'] = json.load(f)
    return metadata


@lru_cache(maxsize=None)
def load_yield_predictor(path='ai/models/yield_prediction/yield_predictor.pkl'):
    resolved_path = _resolve_path(path)
    if resolved_path:
        h5_path = resolved_path.with_suffix('.h5')
        if h5_path.exists():
            return KerasTabularModelWrapper(h5_path, resolved_path.parent, is_regression=True)
        if resolved_path.exists():
            return joblib.load(resolved_path)
    return None


@lru_cache(maxsize=None)
def load_yield_metadata(dir_path='ai/models/yield_prediction'):
    metadata = {}
    resolved_dir = _resolve_path(dir_path)
    if resolved_dir is None:
        return metadata
    feature_path = resolved_dir / 'feature_columns.json'
    if feature_path.exists():
        with open(feature_path, 'r', encoding='utf-8') as f:
            metadata['feature_columns'] = json.load(f)
    return metadata


@lru_cache(maxsize=None)
def load_price_predictor(path='ai/models/price_prediction/price_predictor.pkl'):
    resolved_path = _resolve_path(path)
    if resolved_path:
        h5_path = resolved_path.with_suffix('.h5')
        if h5_path.exists():
            return KerasTabularModelWrapper(h5_path, resolved_path.parent, is_regression=True)
        if resolved_path.exists():
            return joblib.load(resolved_path)
    return None


@lru_cache(maxsize=None)
def load_price_metadata(dir_path='ai/models/price_prediction'):
    metadata = {}
    resolved_dir = _resolve_path(dir_path)
    if resolved_dir is None:
        return metadata
    feature_path = resolved_dir / 'feature_columns.json'
    if feature_path.exists():
        with open(feature_path, 'r', encoding='utf-8') as f:
            metadata['feature_columns'] = json.load(f)
    return metadata


@lru_cache(maxsize=None)
def load_fertilizer_recommender(path='ai/models/fertilizer_recommendation/fertilizer_recommender.pkl'):
    resolved_path = _resolve_path(path)
    if resolved_path:
        h5_path = resolved_path.with_suffix('.h5')
        if h5_path.exists():
            return KerasTabularModelWrapper(h5_path, resolved_path.parent, is_regression=False)
        if resolved_path.exists():
            return joblib.load(resolved_path)
    return None


@lru_cache(maxsize=None)
def load_fertilizer_metadata(dir_path='ai/models/fertilizer_recommendation'):
    metadata = {}
    resolved_dir = _resolve_path(dir_path)
    if resolved_dir is None:
        return metadata
    feature_path = resolved_dir / 'feature_columns.json'
    encoder_path = resolved_dir / 'label_encoders.json'
    label_path = resolved_dir / 'label_classes.json'
    if feature_path.exists():
        with open(feature_path, 'r', encoding='utf-8') as f:
            metadata['feature_columns'] = json.load(f)
    if encoder_path.exists():
        with open(encoder_path, 'r', encoding='utf-8') as f:
            metadata['label_encoders'] = json.load(f)
    if label_path.exists():
        with open(label_path, 'r', encoding='utf-8') as f:
            metadata['label_classes'] = json.load(f)
    return metadata


@lru_cache(maxsize=None)
def load_labels(path='plant_labels.json'):
    resolved = _resolve_path(path)
    if not resolved or not resolved.exists():
        fallback = _resolve_path(f"ai/models/image_classification/{path}")
        if fallback and fallback.exists():
            resolved = fallback
    if resolved and resolved.exists():
        with open(resolved, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and "classes" in data:
                return data["classes"]
            elif isinstance(data, list):
                return data
    return None


@lru_cache(maxsize=None)
def load_image_model(path='ai/models/image_classification/plant_disease_model.pth', device=None):
    resolved_path = _resolve_path(path)
    if not resolved_path or not resolved_path.exists():
        # Fallbacks across known naming conventions
        filename = Path(path).name
        candidates = [
            path,
            filename,
            f"ai/models/image_classification/{filename}",
            'ai/models/image_classification/animal_disease_resnet9.pth',
            'animal_disease_resnet9.pth',
            'ai/models/image_classification/animal_disease_model.pth',
            'animal_disease_model.pth',
            'ai/models/image_classification/plant_disease_model.pth',
            'plant_disease_model.pth',
            'ai/models/image_classification/animal_resnet9_best.pt',
            'ai/models/image_classification/plant_resnet9_best.pt',
            'ai/models/image_classification/animal_resnet9.pt',
            'ai/models/image_classification/plant_resnet9.pt'
        ]
        for cand in candidates:
            cand_res = _resolve_path(cand)
            if cand_res and cand_res.exists():
                resolved_path = cand_res
                break
                
    if not resolved_path or not resolved_path.exists():
        return None
        
    # Check if a Keras .h5 file exists for this model base name
    h5_path = resolved_path.with_suffix('.h5')
    if h5_path.exists():
        try:
            import tensorflow as tf
            try:
                from ai.augmentation import GaussianNoiseLayer, ColorAugmentationLayer
                custom_objects = {
                    'GaussianNoiseLayer': GaussianNoiseLayer,
                    'ColorAugmentationLayer': ColorAugmentationLayer
                }
            except Exception:
                custom_objects = {}
                
            model = tf.keras.models.load_model(h5_path, custom_objects=custom_objects)
            
            # Load classes
            classes_path = h5_path.parent / "label_classes.json"
            if classes_path.exists():
                with open(classes_path, 'r', encoding='utf-8') as f:
                    classes = json.load(f)
            else:
                classes = []
                
            return {
                'model_type': 'keras',
                'model': model,
                'classes': classes
            }
        except Exception as e:
            pass

    # PyTorch checkpoint loading
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    
    # Normalize dictionary structure
    if isinstance(checkpoint, dict):
        if 'model_state' not in checkpoint and 'model_state_dict' in checkpoint:
            checkpoint['model_state'] = checkpoint['model_state_dict']
            
        if 'classes' not in checkpoint:
            parent = resolved_path.parent
            if 'plant' in resolved_path.stem.lower():
                labels = load_labels('plant_labels.json')
            elif 'animal' in resolved_path.stem.lower() or 'livestock' in resolved_path.stem.lower():
                labels = load_labels('animal_labels.json') or load_labels('animal_class_names.json')
            else:
                labels = load_labels(parent / 'labels.json')
                
            if labels:
                checkpoint['classes'] = labels
                
    return checkpoint


def load_plant_disease_model(device=None):
    return load_image_model('ai/models/image_classification/plant_disease_model.pth', device=device)


def load_animal_disease_model(device=None):
    return load_image_model('ai/models/image_classification/animal_disease_resnet9.pth', device=device)
