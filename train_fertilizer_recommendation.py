"""
Fertilizer Recommendation Model Training Pipeline
Task: Fertilizer Recommendation
Model: XGBoost Classifier
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier


def main():
    print("=" * 60)
    print("TASK 2: FERTILIZER RECOMMENDATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1 — EXPLORE
    # ---------------------------------------------------------
    data_dir = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\fertilizer_prediction"
    files = os.listdir(data_dir)
    print(f"\n[STEP 1] Listing files in '{data_dir}':")
    for f in files:
        fpath = os.path.join(data_dir, f)
        print(f"  - {f} (size: {os.path.getsize(fpath)} bytes)")

    primary_file = "Fertilizer Prediction.csv"
    selected_path = os.path.join(data_dir, primary_file)
    print(f"\nLoading '{primary_file}'...")

    df = pd.read_csv(selected_path)
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Standardize column naming
    rename_dict = {
        "Temparature": "temperature",
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Moisture": "moisture",
        "Soil Type": "soil_type",
        "Crop Type": "crop_type",
        "Nitrogen": "nitrogen",
        "Potassium": "potassium",
        "Phosphorous": "phosphorous",
        "Fertilizer Name": "fertilizer_name"
    }
    df = df.rename(columns=rename_dict)

    print("\nDataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nNull Value Counts:")
    print(df.isnull().sum())

    target_col = "fertilizer_name"
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"\nIdentified Target Column: '{target_col}'")
    print(f"Identified Feature Columns ({len(feature_cols)}): {feature_cols}")

    print("\nTarget Class Distribution:")
    class_counts = df[target_col].value_counts()
    print(class_counts)
    print(f"Total Unique Classes: {len(class_counts)}")

    # ---------------------------------------------------------
    # STEP 2 — PREPROCESS
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[STEP 2] PREPROCESSING & CATEGORICAL ENCODING")
    print("-" * 60)

    # Missing values
    if df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    else:
        print("No missing values detected.")

    # Encode categorical feature columns
    categorical_feature_cols = ["soil_type", "crop_type"]
    feature_encoders = {}
    X = df[feature_cols].copy()

    for col in categorical_feature_cols:
        le_feat = LabelEncoder()
        X[col] = le_feat.fit_transform(X[col].astype(str))
        feature_encoders[col] = le_feat
        print(f"Encoded '{col}': {list(le_feat.classes_)}")

    # Encode target column
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(df[target_col])
    print(f"\nEncoded Target '{target_col}': {list(target_encoder.classes_)}")

    # Train/Test Split (80/20, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # ---------------------------------------------------------
    # STEP 3 — TRAIN & HYPERPARAMETER TUNING
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[STEP 3] MODEL TRAINING & HYPERPARAMETER TUNING")
    print("-" * 60)

    xgb_base = XGBClassifier(
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False
    )

    param_grid = {
        'n_estimators': [30, 50, 100],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.05, 0.1, 0.2]
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest Hyperparameters: {grid_search.best_params_}")
    print(f"Best 5-Fold CV Accuracy: {grid_search.best_score_:.4f}")

    # ---------------------------------------------------------
    # STEP 4 — EVALUATE
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[STEP 4] EVALUATION ON TEST SET")
    print("-" * 60)

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1-Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_, zero_division=0))

    print("Confusion Matrix:")
    print(cm)

    # Feature Importance
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    for feat, imp in feat_imp:
        print(f"  - {feat:15s}: {imp:.4f}")

    # ---------------------------------------------------------
    # STEP 5 — SAVE ARTIFACTS
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[STEP 5] SAVING ARTIFACTS")
    print("-" * 60)

    models_dir = r"C:\Users\Dhanush\OneDrive\Desktop\CL\models"
    os.makedirs(models_dir, exist_ok=True)
    fert_sub_dir = os.path.join(models_dir, "fertilizer_recommendation")
    os.makedirs(fert_sub_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "fertilizer_recommendation_model.pkl")
    target_encoder_path = os.path.join(models_dir, "fertilizer_label_encoder.pkl")
    feature_encoders_path = os.path.join(models_dir, "fertilizer_feature_encoders.pkl")
    feature_order_path = os.path.join(models_dir, "fertilizer_feature_order.json")

    joblib.dump(best_model, model_path)
    joblib.dump(best_model, os.path.join(fert_sub_dir, "fertilizer_recommendation_model.pkl"))
    joblib.dump(target_encoder, target_encoder_path)
    joblib.dump(target_encoder, os.path.join(fert_sub_dir, "fertilizer_label_encoder.pkl"))
    joblib.dump(feature_encoders, feature_encoders_path)
    joblib.dump(feature_encoders, os.path.join(fert_sub_dir, "fertilizer_feature_encoders.pkl"))

    feature_order_data = {
        "task": "fertilizer_recommendation",
        "model_type": "XGBClassifier",
        "feature_order": feature_cols,
        "target_col": target_col,
        "classes": target_encoder.classes_.tolist(),
        "categorical_encodings": {
            col: list(enc.classes_) for col, enc in feature_encoders.items()
        }
    }
    with open(feature_order_path, "w") as f:
        json.dump(feature_order_data, f, indent=4)
    with open(os.path.join(fert_sub_dir, "feature_order.json"), "w") as f:
        json.dump(feature_order_data, f, indent=4)

    print(f"Saved model: {model_path}")
    print(f"Saved target label encoder: {target_encoder_path}")
    print(f"Saved feature encoders: {feature_encoders_path}")
    print(f"Saved feature order: {feature_order_path}")
    print("\nTask 2 Fertilizer Recommendation completed successfully.")

    return {
        "task": "Fertilizer Recommendation",
        "model_type": "XGBoost Classifier",
        "key_metric": f"Accuracy: {acc:.4f} (F1: {f1:.4f})",
        "saved_files": f"{model_path}, {target_encoder_path}, {feature_encoders_path}, {feature_order_path}"
    }


if __name__ == "__main__":
    main()
