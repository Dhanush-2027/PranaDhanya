import os
import json
import joblib
import torch

def load_crop_recommender(path='ai/models/crop_recommendation/crop_recommender.pkl'):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_yield_predictor(path='ai/models/yield_prediction/yield_predictor.pkl'):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_price_predictor(path='ai/models/price_prediction/price_predictor.pkl'):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_fertilizer_recommender(path='ai/models/fertilizer_recommendation/fertilizer_recommender.pkl'):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_fertilizer_metadata(dir_path='ai/models/fertilizer_recommendation'):
    metadata = {}
    feature_path = os.path.join(dir_path, 'feature_columns.json')
    encoder_path = os.path.join(dir_path, 'label_encoders.json')
    label_path = os.path.join(dir_path, 'label_classes.json')
    if os.path.exists(feature_path):
        with open(feature_path, 'r', encoding='utf-8') as f:
            metadata['feature_columns'] = json.load(f)
    if os.path.exists(encoder_path):
        with open(encoder_path, 'r', encoding='utf-8') as f:
            metadata['label_encoders'] = json.load(f)
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            metadata['label_classes'] = json.load(f)
    return metadata


def load_image_model(path='ai/models/image_classification/resnet9.pt', device=None):
    if not os.path.exists(path):
        return None
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(path, map_location=device)
    return checkpoint
