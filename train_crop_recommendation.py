"""
Crop Recommendation Model Training Pipeline
Task: Crop Recommendation
Model: XGBoost Classifier
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    print("TASK 1: CROP RECOMMENDATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1 — EXPLORE
    # ---------------------------------------------------------
    data_dir = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\crop_recommendation"
    files = os.listdir(data_dir)
    print(f"\n[STEP 1] Listing files in '{data_dir}':")
    for f in files:
        fpath = os.path.join(data_dir, f)
        print(f"  - {f} (size: {os.path.getsize(fpath)} bytes)")

    # Selecting the primary full dataset
    primary_file = "Crop_recommendation.csv"
    selected_path = os.path.join(data_dir, primary_file)
    print(f"\nUsing '{primary_file}' because it is the full, complete dataset (2200 samples) compared to data.csv (stub with 3 rows).")

    df = pd.read_csv(selected_path)
    print("\nDataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nNull Value Counts:")
    print(df.isnull().sum())

    target_col = "label"
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
    print("[STEP 2] PREPROCESSING")
    print("-" * 60)

    # Missing value handling
    if df.isnull().sum().sum() > 0:
        print("Imputing missing values with median...")
        for col in feature_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    else:
        print("No missing values detected.")

    # Check for outliers / statistics
    print("\nSummary Statistics:")
    print(df[feature_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])

    # Encode categorical target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(df[target_col])
    X = df[feature_cols].copy()

    # Split into train/test (80/20, random_state=42, stratify=y)
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
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2]
    }

    print("Running GridSearchCV with 5-fold cross validation...")
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=5,
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
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

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
    crop_sub_dir = os.path.join(models_dir, "crop_recommendation")
    os.makedirs(crop_sub_dir, exist_ok=True)

    # Save root models folder and subfolder for convenience
    model_path = os.path.join(models_dir, "crop_recommendation_model.pkl")
    encoder_path = os.path.join(models_dir, "crop_label_encoder.pkl")
    feature_order_path = os.path.join(models_dir, "crop_feature_order.json")

    joblib.dump(best_model, model_path)
    joblib.dump(best_model, os.path.join(crop_sub_dir, "crop_recommendation_model.pkl"))
    joblib.dump(target_encoder, encoder_path)
    joblib.dump(target_encoder, os.path.join(crop_sub_dir, "crop_label_encoder.pkl"))

    feature_order_data = {
        "task": "crop_recommendation",
        "model_type": "XGBClassifier",
        "feature_order": feature_cols,
        "target_col": target_col,
        "classes": target_encoder.classes_.tolist()
    }
    with open(feature_order_path, "w") as f:
        json.dump(feature_order_data, f, indent=4)
    with open(os.path.join(crop_sub_dir, "feature_order.json"), "w") as f:
        json.dump(feature_order_data, f, indent=4)

    print(f"Saved model: {model_path}")
    print(f"Saved label encoder: {encoder_path}")
    print(f"Saved feature order: {feature_order_path}")
    print("\nTask 1 Crop Recommendation completed successfully.")

    return {
        "task": "Crop Recommendation",
        "model_type": "XGBoost Classifier",
        "key_metric": f"Accuracy: {acc:.4f} (F1: {f1:.4f})",
        "saved_files": f"{model_path}, {encoder_path}, {feature_order_path}"
    }


if __name__ == "__main__":
    main()
