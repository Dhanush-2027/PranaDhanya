import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
)
from ai.utils import logger, plot_confusion_matrix, plot_roc_curve

def evaluate_classification_model(model, X_test, y_test, class_names, output_dir, name):
    """
    Computes classification performance metrics:
    Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC.
    Saves metrics as JSON and plots as PNGs.
    """
    logger.info(f"Evaluating classification model: {name}")
    
    # 1. Get predictions
    if isinstance(X_test, tf.data.Dataset):
        y_pred_probs = model.predict(X_test)
    else:
        y_pred_probs = model.predict(X_test)
        
    n_classes = len(class_names)
    
    # If binary classification and output shape is (N, 1)
    if y_pred_probs.shape[-1] == 1:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_pred_probs_onehot = np.hstack([1 - y_pred_probs, y_pred_probs])
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_pred_probs_onehot = y_pred_probs
        
    y_test = np.array(y_test)
    
    # 2. Basic metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # 3. ROC-AUC (Multi-class One-vs-Rest or Binary)
    auc_score = 0.5
    try:
        if n_classes == 2:
            auc_score = roc_auc_score(y_test, y_pred_probs_onehot[:, 1])
        else:
            # One-hot encode y_test for multi-class AUC
            y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
            auc_score = roc_auc_score(y_test_onehot, y_pred_probs_onehot, multi_class='ovr', average='weighted')
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        
    # 4. Top-5 Accuracy (if applicable)
    top5_acc = 0.0
    if n_classes >= 5:
        try:
            # Sort top 5 indices per row
            top5_preds = np.argsort(y_pred_probs_onehot, axis=1)[:, -5:]
            matches = [y_test[i] in top5_preds[i] for i in range(len(y_test))]
            top5_acc = np.mean(matches)
        except Exception:
            pass
            
    # 5. Generate plots
    # Convert labels to one-hot for plotting
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
    plot_confusion_matrix(y_test, y_pred, class_names, output_dir, name)
    plot_roc_curve(y_test_onehot, y_pred_probs_onehot, class_names, output_dir, name)
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(auc_score),
        "top5_accuracy": float(top5_acc) if n_classes >= 5 else None
    }
    
    # Save metrics JSON
    with open(output_dir / "metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Classification Metrics for {name}: {metrics}")
    return metrics

def evaluate_regression_model(model, X_test, y_test, output_dir, name):
    """
    Computes regression performance metrics:
    MAE, MSE, RMSE, R² Score.
    Saves metrics as JSON.
    """
    logger.info(f"Evaluating regression model: {name}")
    
    y_pred = model.predict(X_test).flatten()
    y_test = np.array(y_test).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2)
    }
    
    with open(output_dir / "metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Regression Metrics for {name}: {metrics}")
    return metrics
