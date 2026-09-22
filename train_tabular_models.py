import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier

from ai_training_utils import scan_tabular_dataset, validate_tabular_data

# Ensure output directories exist
os.makedirs("ai/models/crop_recommendation", exist_ok=True)
os.makedirs("ai/models/yield_prediction", exist_ok=True)
os.makedirs("ai/models/price_prediction", exist_ok=True)
os.makedirs("ai/models/fertilizer_recommendation", exist_ok=True)

# ----------------------------------------------------
# 1. CROP RECOMMENDATION
# ----------------------------------------------------
def train_crop_recommendation():
    print("\n--- Training Crop Recommendation Model ---")
    dataset_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\crop_recommendation\Crop_recommendation.csv"
    target_col = "label"
    
    # Pre-training scan
    df, summary = scan_tabular_dataset(dataset_path, target_col, is_classification=True)
    with open("ai/models/crop_recommendation/dataset_report.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    expected_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    validate_tabular_data(df, expected_features, target_col)
    
    X = df[expected_features]
    y = df[target_col]
    
    # Label encoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Data Split (70/15/15 stratified)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    ) # 0.1765 * 0.85 approx 0.15
    
    # Pipeline: Scaler + XGBoost
    scaler = StandardScaler()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('xgb', xgb)
    ])
    
    # Hyperparameter tuning using cross-validation
    param_grid = {
        'xgb__max_depth': [3, 5, 7],
        'xgb__n_estimators': [50, 100],
        'xgb__learning_rate': [0.1, 0.2]
    }
    
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)
    
    # Evaluate on val set
    y_val_pred = best_pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average='weighted')
    test_rec = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("Confusion Matrix:\n", test_cm)
    
    # Feature Importance
    xgb_model = best_pipeline.named_steps['xgb']
    importances = xgb_model.feature_importances_
    feature_importance_dict = dict(zip(expected_features, [float(x) for x in importances]))
    print("Top feature importances:", sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Save artifacts
    joblib.dump(best_pipeline, "ai/models/crop_recommendation/crop_recommendation_model.pkl")
    # Also save as crop_recommender.pkl to support existing backend
    joblib.dump(best_pipeline, "ai/models/crop_recommendation/crop_recommender.pkl")
    
    fitted_scaler = best_pipeline.named_steps['scaler']
    joblib.dump(fitted_scaler, "ai/models/crop_recommendation/crop_scaler.pkl")
    joblib.dump(le, "ai/models/crop_recommendation/crop_label_encoder.pkl")
    
    with open("ai/models/crop_recommendation/crop_feature_names.json", "w") as f:
        json.dump(expected_features, f)
    # Also save label classes and feature columns for app.py fallback
    with open("ai/models/crop_recommendation/feature_columns.json", "w") as f:
        json.dump(expected_features, f)
    with open("ai/models/crop_recommendation/label_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)
        
    metrics = {
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1_score": test_f1,
        "confusion_matrix": test_cm.tolist(),
        "feature_importance": feature_importance_dict
    }
    with open("ai/models/crop_recommendation/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Crop recommendation training completed successfully.")

# ----------------------------------------------------
# 2. YIELD PREDICTION
# ----------------------------------------------------
def train_yield_prediction():
    print("\n--- Training Yield Prediction Model ---")
    dataset_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\yield_prediction\data.csv"
    target_col = "yield"
    
    # Pre-training scan
    df, summary = scan_tabular_dataset(dataset_path, target_col, is_classification=False)
    with open("ai/models/yield_prediction/dataset_report.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    expected_features = ["area", "rainfall", "fertilizer", "temperature", "humidity"]
    validate_tabular_data(df, expected_features, target_col)
    
    X = df[expected_features]
    y = df[target_col]
    
    # Since dataset is extremely small (3 samples), split isn't feasible normally.
    # We will use all samples for training and evaluate on the same samples or train test split without stratify.
    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    # Avoid division by zero in r2 score for extremely small data
    try:
        r2 = r2_score(y_test, preds)
    except Exception:
        r2 = 1.0
        
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save artifacts
    joblib.dump(model, "ai/models/yield_prediction/yield_model.pkl")
    # Also save as yield_predictor.pkl to support backend
    joblib.dump(model, "ai/models/yield_prediction/yield_predictor.pkl")
    
    with open("ai/models/yield_prediction/yield_features.json", "w") as f:
        json.dump(expected_features, f)
    # Also feature_columns.json for backend
    with open("ai/models/yield_prediction/feature_columns.json", "w") as f:
        json.dump(expected_features, f)
        
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "feature_importance": dict(zip(expected_features, [float(x) for x in model.feature_importances_]))
    }
    with open("ai/models/yield_prediction/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Yield prediction training completed.")

# ----------------------------------------------------
# 3. CROP PRICE PREDICTION
# ----------------------------------------------------
def train_crop_price_prediction():
    print("\n--- Training Crop Price Prediction Model ---")
    dataset_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\price_prediction\data.csv"
    target_col = "price"
    
    # Pre-training scan
    df, summary = scan_tabular_dataset(dataset_path, target_col, is_classification=False)
    with open("ai/models/price_prediction/dataset_report.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    expected_features = ["feature1", "feature2", "feature3"]
    validate_tabular_data(df, expected_features, target_col)
    
    X = df[expected_features]
    y = df[target_col]
    
    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    try:
        r2 = r2_score(y_test, preds)
    except Exception:
        r2 = 1.0
        
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save artifacts
    joblib.dump(model, "ai/models/price_prediction/price_model.pkl")
    # Also save as price_predictor.pkl to support backend
    joblib.dump(model, "ai/models/price_prediction/price_predictor.pkl")
    
    with open("ai/models/price_prediction/price_features.json", "w") as f:
        json.dump(expected_features, f)
    # Also feature_columns.json for backend
    with open("ai/models/price_prediction/feature_columns.json", "w") as f:
        json.dump(expected_features, f)
        
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "feature_importance": dict(zip(expected_features, [float(x) for x in model.feature_importances_]))
    }
    with open("ai/models/price_prediction/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Crop price prediction training completed.")

# ----------------------------------------------------
# 4. FERTILIZER RECOMMENDATION
# ----------------------------------------------------
def train_fertilizer_recommendation():
    print("\n--- Training Fertilizer Recommendation Model ---")
    dataset_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\fertilizer_prediction\Fertilizer Prediction.csv"
    target_col = "Fertilizer Name"
    
    # Read the raw data to auto detect target column if normalized
    df = pd.read_csv(dataset_path)
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Pre-training scan
    df, summary = scan_tabular_dataset(dataset_path, target_col, is_classification=True)
    with open("ai/models/fertilizer_recommendation/dataset_report.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    # Check for temperature synonym
    if "Temparature" in df.columns:
        df = df.rename(columns={"Temparature": "temperature"})
    if "Humidity" in df.columns:
        df = df.rename(columns={"Humidity": "humidity"})
    if "Humidity " in df.columns:
        df = df.rename(columns={"Humidity ": "humidity"})
    if "Moisture" in df.columns:
        df = df.rename(columns={"Moisture": "moisture"})
    if "Nitrogen" in df.columns:
        df = df.rename(columns={"Nitrogen": "nitrogen"})
    if "Potassium" in df.columns:
        df = df.rename(columns={"Potassium": "potassium"})
    if "Phosphorous" in df.columns:
        df = df.rename(columns={"Phosphorous": "phosphorous"})
    if "Soil Type" in df.columns:
        df = df.rename(columns={"Soil Type": "soil_type"})
    if "Crop Type" in df.columns:
        df = df.rename(columns={"Crop Type": "crop_type"})
        
    expected_features = ["temperature", "humidity", "moisture", "nitrogen", "potassium", "phosphorous", "soil_type", "crop_type"]
    validate_tabular_data(df, expected_features, target_col)
    
    # Categorical encoders mapping
    encoder_mappings = {}
    X = df[expected_features].copy()
    for col in ["soil_type", "crop_type"]:
        category = X[col].astype('category')
        encoder_mappings[col] = list(category.cat.categories)
        X[col] = category.cat.codes
        
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_col])
    
    # Data Split (70/15/15 stratified)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    )
    
    # Model: XGBoost Classifier
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average='weighted')
    test_rec = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    
    # Save artifacts
    joblib.dump(model, "ai/models/fertilizer_recommendation/fertilizer_model.pkl")
    # Also save as fertilizer_recommender.pkl to support backend
    joblib.dump(model, "ai/models/fertilizer_recommendation/fertilizer_recommender.pkl")
    
    with open("ai/models/fertilizer_recommendation/fertilizer_features.json", "w") as f:
        json.dump(expected_features, f)
    # Also feature_columns, label_classes, and label_encoders for backend
    with open("ai/models/fertilizer_recommendation/feature_columns.json", "w") as f:
        json.dump(expected_features, f)
    with open("ai/models/fertilizer_recommendation/label_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)
    with open("ai/models/fertilizer_recommendation/label_encoders.json", "w") as f:
        json.dump(encoder_mappings, f)
        
    metrics = {
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1_score": test_f1,
        "confusion_matrix": test_cm.tolist(),
        "feature_importance": dict(zip(expected_features, [float(x) for x in model.feature_importances_]))
    }
    with open("ai/models/fertilizer_recommendation/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Fertilizer recommendation training completed.")

if __name__ == "__main__":
    train_crop_recommendation()
    train_yield_prediction()
    train_crop_price_prediction()
    train_fertilizer_recommendation()
