import os
import joblib
import json
from typing import List

def save_sklearn_model(model, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pkl")
    joblib.dump(model, path)
    return path

def load_sklearn_model(path: str):
    return joblib.load(path)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)
