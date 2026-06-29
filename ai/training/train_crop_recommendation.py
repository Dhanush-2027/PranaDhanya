import argparse
import json
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def main(args):
    print(f"Loading dataset from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    # encode labels to 0..n-1 to satisfy XGBoost requirements
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # avoid splitting when dataset is too small (which may drop classes from training)
    n_samples = len(y_encoded)
    if n_samples > len(set(y_encoded)) and n_samples >= 5:
        X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    else:
        X_train, y_train = X, y_encoded
        X_val, y_val = None, None
    
    # adapt objective for multiclass if necessary
    unique_classes = sorted(set(y_encoded))
    if len(unique_classes) > 2:
        model = XGBClassifier(use_label_encoder=False, objective='multi:softprob', num_class=len(unique_classes), eval_metric='mlogloss', random_state=42, n_estimators=100)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=100)
    
    print(f"Training XGBoost classifier with {len(unique_classes)} classes")
    model.fit(X_train, y_train)
    
    # Evaluate on training data
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training accuracy: {train_acc:.4f}")
    
    if X_val is not None:
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"Validation accuracy: {val_acc:.4f}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.out_dir, 'crop_recommender.pkl'))
    with open(os.path.join(args.out_dir, 'feature_columns.json'), 'w') as f:
        json.dump(list(X.columns), f)
    # save label classes for decoding predictions
    with open(os.path.join(args.out_dir, 'label_classes.json'), 'w') as f:
        json.dump(list(le.classes_.tolist()), f)
    
    print('✓ Saved crop_recommender.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV file with features and target')
    parser.add_argument('--target', default='label', help='Target column name')
    parser.add_argument('--out-dir', default='ai/models/crop_recommendation')
    args = parser.parse_args()
    main(args)
