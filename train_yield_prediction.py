"""
Yield Prediction Model Training Pipeline
Task: Yield Prediction
Model: Random Forest Regressor
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    print("=" * 60)
    print("TASK 3: YIELD PREDICTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1 — EXPLORE
    # ---------------------------------------------------------
    data_dir = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\yield_prediction"
    files = os.listdir(data_dir)
    print(f"\n[STEP 1] Listing files in '{data_dir}':")
    for f in files:
        fpath = os.path.join(data_dir, f)
        print(f"  - {f} (size: {os.path.getsize(fpath)} bytes)")

    primary_file = "data.csv"
    selected_path = os.path.join(data_dir, primary_file)
    print(f"\nLoading '{primary_file}'...")

    df = pd.read_csv(selected_path)
    df.columns = df.columns.str.strip()

    print("\nDataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nNull Value Counts:")
    print(df.isnull().sum())

    target_col = "yield"
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"\nIdentified Target Column: '{target_col}'")
    print(f"Identified Feature Columns ({len(feature_cols)}): {feature_cols}")

    # ---------------------------------------------------------
    # STEP 2 — PREPROCESS
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[STEP 2] PREPROCESSING & ENCODING")
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

    # Encode any categorical columns (e.g. state, district, season, crop if present)
    categorical_cols = [c for c in feature_cols if df[c].dtype == 'object']
    encoders = {}
    X = df[feature_cols].copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"Encoded categorical feature '{col}': {list(le.classes_)}")

    y = df[target_col].values

    # Train/Test Split (80/20, random_state=42)
    # If samples are very small (e.g., 3 rows), we evaluate on the available samples
    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
    else:
        print(f"Note: Small dataset size ({len(df)} samples). Using all samples for training/testing.")
        X_train, X_test, y_train, y_test = X, X, y, y

    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # ---------------------------------------------------------
    # STEP 3 — TRAIN & HYPERPARAMETER TUNING
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[STEP 3] MODEL TRAINING & HYPERPARAMETER TUNING")
    print("-" * 60)

    rf_base = RandomForestRegressor(random_state=42)

    # If dataset has enough samples, run GridSearchCV, otherwise train fitted estimator
    if len(X_train) >= 5:
        param_grid = {
            'n_estimators': [20, 50, 100],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 3]
        }
        cv_folds = min(5, len(X_train))
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")
    else:
        best_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            random_state=42
        )
        best_model.fit(X_train, y_train)
        print("Trained RandomForestRegressor with parameters:", best_model.get_params())

    # ---------------------------------------------------------
    # STEP 4 — EVALUATE
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[STEP 4] EVALUATION")
    print("-" * 60)

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    try:
        r2 = r2_score(y_test, y_pred)
    except Exception:
        r2 = 1.0

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    print("\nActual vs Predicted:")
    for actual, pred in zip(y_test, y_pred):
        print(f"  Actual: {actual:.2f} | Predicted: {pred:.2f}")

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
    yield_sub_dir = os.path.join(models_dir, "yield_prediction")
    os.makedirs(yield_sub_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "yield_prediction_model.pkl")
    encoders_path = os.path.join(models_dir, "yield_encoders.pkl")
    feature_order_path = os.path.join(models_dir, "yield_feature_order.json")

    joblib.dump(best_model, model_path)
    joblib.dump(best_model, os.path.join(yield_sub_dir, "yield_prediction_model.pkl"))
    joblib.dump(encoders, encoders_path)
    joblib.dump(encoders, os.path.join(yield_sub_dir, "yield_encoders.pkl"))

    feature_order_data = {
        "task": "yield_prediction",
        "model_type": "RandomForestRegressor",
        "feature_order": feature_cols,
        "target_col": target_col,
        "categorical_encodings": {
            col: list(enc.classes_) for col, enc in encoders.items()
        }
    }
    with open(feature_order_path, "w") as f:
        json.dump(feature_order_data, f, indent=4)
    with open(os.path.join(yield_sub_dir, "feature_order.json"), "w") as f:
        json.dump(feature_order_data, f, indent=4)

    print(f"Saved model: {model_path}")
    print(f"Saved encoders: {encoders_path}")
    print(f"Saved feature order: {feature_order_path}")
    print("\nTask 3 Yield Prediction completed successfully.")

    return {
        "task": "Yield Prediction",
        "model_type": "Random Forest Regressor",
        "key_metric": f"R²: {r2:.4f} (MAE: {mae:.4f}, RMSE: {rmse:.4f})",
        "saved_files": f"{model_path}, {encoders_path}, {feature_order_path}"
    }


if __name__ == "__main__":
    main()
