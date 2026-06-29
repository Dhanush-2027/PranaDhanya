import argparse
import json
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main(args):
    print(f"Loading dataset from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Random Forest Regressor with {len(X.columns)} features")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    
    # Evaluate on validation data
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    print(f"Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.out_dir, 'yield_predictor.pkl'))
    
    # Save feature metadata
    with open(os.path.join(args.out_dir, 'feature_columns.json'), 'w') as f:
        json.dump(list(X.columns), f)
    
    print('✓ Saved yield_predictor.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--target', default='yield')
    parser.add_argument('--out-dir', default='ai/models/yield_prediction')
    args = parser.parse_args()
    main(args)
