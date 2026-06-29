import argparse
import json
import joblib
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier


def normalize_columns(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('___', '_').str.lower()
    return df


def factorize_series(series):
    labels, uniques = pd.factorize(series.astype(str), sort=True)
    return labels, list(uniques)


def main(args):
    df = pd.read_csv(args.input)
    df = normalize_columns(df)

    # Normalize target column name to match normalized columns
    target_col = args.target.strip().lower().replace(' ', '_').replace('-', '_')
    if target_col not in df.columns:
        raise ValueError(f"Target column '{args.target}' (normalized: '{target_col}') not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_columns = ['temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous', 'soil_type', 'crop_type']
    if 'temparature' in X.columns and 'temperature' not in X.columns:
        X['temperature'] = X['temparature']
    if not set(feature_columns).issubset(set(X.columns)):
        raise ValueError(f"Expected feature columns {feature_columns}, but dataset contains {list(X.columns)}")
    X = X[feature_columns].copy()

    encoder_mappings = {}
    for col in ['soil_type', 'crop_type']:
        if col in X.columns:
            category = X[col].astype('category')
            encoder_mappings[col] = list(category.cat.categories)
            X[col] = category.cat.codes

    # Ensure all training features are numeric before passing to XGBoost
    X = X.astype(float)
    y_encoded, label_classes = factorize_series(y)

    n_samples = len(y_encoded)
    if n_samples > len(set(y_encoded)) and n_samples >= 5:
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_samples)
        split = int(n_samples * 0.8)
        train_idx = indices[:split]
        X_train = X.iloc[train_idx]
        y_train = y_encoded[train_idx]
    else:
        X_train, y_train = X, y_encoded

    unique_classes = sorted(set(y_encoded))
    if len(unique_classes) > 2:
        model = XGBClassifier(
            use_label_encoder=False,
            objective='multi:softprob',
            num_class=len(unique_classes),
            eval_metric='mlogloss',
            random_state=42
        )
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    model.fit(X_train, y_train)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.out_dir, 'fertilizer_recommender.pkl'))
    with open(os.path.join(args.out_dir, 'feature_columns.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f)
    with open(os.path.join(args.out_dir, 'label_classes.json'), 'w', encoding='utf-8') as f:
        json.dump(label_classes, f)
    with open(os.path.join(args.out_dir, 'label_encoders.json'), 'w', encoding='utf-8') as f:
        json.dump(encoder_mappings, f)

    print('Saved fertilizer_recommender.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV file with features and target')
    parser.add_argument('--target', default='fertilizer_name', help='Target column name')
    parser.add_argument('--out-dir', default='ai/models/fertilizer_recommendation')
    args = parser.parse_args()
    # Auto-detect the target column if it doesn't exist
    df = pd.read_csv(args.input)
    normalized_cols = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    if args.target not in df.columns and args.target.lower() in normalized_cols:
        idx = normalized_cols.index(args.target.lower())
        args.target = df.columns[idx]
    main(args)
