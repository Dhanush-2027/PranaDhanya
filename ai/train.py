import os
import sys
import json
import time
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

# Add workspace root to sys.path to enable 'ai.*' imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.config import (
    DATASET_DIR, MODEL_DIR, TENSORBOARD_DIR, RANDOM_SEED,
    MAX_EPOCHS, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)
from ai.utils import (
    logger, set_seeds, detect_device, get_system_usage,
    plot_and_save_curves
)
from ai.preprocessing import (
    preprocess_tabular_dataset, balance_tabular_classes
)
from ai.data_loader import (
    detect_datasets, detect_target_column, split_tabular_data,
    load_image_paths_and_labels, build_image_datasets
)
from ai.models.resnet9_tf import ResNet9TF
from ai.augmentation import get_augmentation_pipeline, GaussianNoiseLayer, ColorAugmentationLayer
from ai.trainer import (
    build_tabular_model, select_best_optimizer,
    tune_hyperparameters, export_model_formats,
    get_dynamic_batch_size, get_optimizer, TrainingStateCallback
)
from ai.evaluation import (
    evaluate_classification_model, evaluate_regression_model
)

def detect_task_type(series):
    """Automatically determine if the task is regression, binary, or multi-class classification."""
    if not pd.api.types.is_numeric_dtype(series):
        if series.nunique() == 2:
            return "binary_classification"
        return "multi_class_classification"
        
    unique_vals = series.dropna().unique()
    if len(unique_vals) <= 15:  # Discrete categories
        if len(unique_vals) == 2:
            return "binary_classification"
        return "multi_class_classification"
        
    return "regression"

def train_tabular_model(dataset_name, csv_paths, max_epochs=MAX_EPOCHS):
    """Orchestrates loading, preprocessing, tuning, training, and exporting of a tabular model."""
    logger.info(f"\n{'='*70}\n>>> Training Tabular Model: {dataset_name}\n{'='*70}")
    
    # Use the first/largest CSV file found
    csv_path = csv_paths[0]
    df = pd.read_csv(csv_path)
    
    output_dir = MODEL_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Detect target and task type
    target_col = detect_target_column(df, dataset_name)
    task_type = detect_task_type(df[target_col])
    is_classification = (task_type != "regression")
    logger.info(f"Target column: {target_col} | Task: {task_type}")
    
    # 2. Preprocess
    df_processed, label_encoders, target_encoder, scaler, feature_names = preprocess_tabular_dataset(
        df, target_col, output_dir, is_classification=is_classification
    )
    
    # 3. Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_tabular_data(
        df_processed, target_col, is_classification=is_classification
    )
    
    # 4. Class Balancing (Oversampling)
    if is_classification:
        X_train, y_train = balance_tabular_classes(X_train, y_train)
        num_classes = len(target_encoder.classes_)
    else:
        num_classes = 1
        
    # Convert features to float arrays
    X_train_arr = X_train.values.astype(float)
    X_val_arr = X_val.values.astype(float)
    X_test_arr = X_test.values.astype(float)
    y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    # 5. Hyperparameter Tuning
    build_wrapper = lambda hp=None: build_tabular_model(len(feature_names), num_classes, task_type, hp)
    best_hp = tune_hyperparameters(
        build_wrapper, (X_train_arr, y_train_arr), (X_val_arr, y_val_arr), task_type, len(feature_names)
    )
    
    # 6. Optimizer Selection
    build_tuned_wrapper = lambda hp=None: build_tabular_model(len(feature_names), num_classes, task_type, best_hp)
    best_opt = select_best_optimizer(
        build_tuned_wrapper, (X_train_arr, y_train_arr), (X_val_arr, y_val_arr), epochs=3
    )
    
    # 7. Check for Interrupted Training (Resume)
    checkpoint_file = output_dir / "last_checkpoint.h5"
    state_file = output_dir / "training_state.json"
    initial_epoch = 0
    
    # Setup loss and metrics
    if task_type == "regression":
        loss = "mse"
        metrics = ["mae", "mse"]
    elif task_type == "binary_classification":
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
        
    if checkpoint_file.exists() and state_file.exists():
        try:
            logger.info("Resuming training from last saved checkpoint...")
            model = tf.keras.models.load_model(checkpoint_file)
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            initial_epoch = state.get("epoch", 0)
            logger.info(f"Resuming at epoch {initial_epoch + 1}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Building new model.")
            model = build_tuned_wrapper()
            model.compile(optimizer=get_optimizer(best_opt, best_hp.get("lr", 1e-3)), loss=loss, metrics=metrics)
    else:
        model = build_tuned_wrapper()
        model.compile(optimizer=get_optimizer(best_opt, best_hp.get("lr", 1e-3)), loss=loss, metrics=metrics)
        
    # 8. Setup Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(output_dir / "best_model.h5"), monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(TENSORBOARD_DIR / dataset_name), histogram_freq=1),
        TrainingStateCallback(output_dir)
    ]
    
    # 9. Train Model
    batch_size = get_dynamic_batch_size(is_image_dataset=False)
    logger.info(f"Training parameters: batch_size={batch_size} | max_epochs={max_epochs}")
    
    start_time = time.time()
    history = model.fit(
        X_train_arr, y_train_arr,
        validation_data=(X_val_arr, y_val_arr),
        epochs=max_epochs,
        batch_size=batch_size,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds.")
    
    # 10. Save plots
    plot_and_save_curves(history, output_dir, dataset_name)
    
    # 11. Load Best Weights and Evaluate
    if (output_dir / "best_model.h5").exists():
        logger.info("Loading best model weights for evaluation...")
        model = tf.keras.models.load_model(output_dir / "best_model.h5")
        
    if is_classification:
        class_names = list(target_encoder.classes_)
        metrics_results = evaluate_classification_model(
            model, X_test_arr, y_test_arr, class_names, output_dir, dataset_name
        )
    else:
        metrics_results = evaluate_regression_model(
            model, X_test_arr, y_test_arr, output_dir, dataset_name
        )
        
    # 12. Export Formats
    export_model_formats(model, output_dir, "model", (len(feature_names),))
    
    # 13. Backwards Compatibility Copy
    handle_backwards_compatibility_copy(dataset_name, output_dir)
    
    # Clean up checkpoint files once training completes cleanly
    try:
        if checkpoint_file.exists(): os.remove(checkpoint_file)
        if state_file.exists(): os.remove(state_file)
    except Exception:
        pass
        
    return metrics_results

def train_image_model(dataset_name, data_dir, max_epochs=MAX_EPOCHS, max_samples=None):
    """Orchestrates image loading, ResNet9 setup, training, and exporting."""
    logger.info(f"\n{'='*70}\n>>> Training Image Model: {dataset_name}\n{'='*70}")
    
    output_dir = MODEL_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load paths & labels
    image_paths, labels, class_names = load_image_paths_and_labels(data_dir)
    
    # Subsample if max_samples is provided
    if max_samples is not None:
        class_samples = {}
        for path, label in zip(image_paths, labels):
            if label not in class_samples:
                class_samples[label] = []
            if len(class_samples[label]) < max_samples:
                class_samples[label].append(path)
                
        image_paths = []
        labels = []
        for label, paths in class_samples.items():
            image_paths.extend(paths)
            labels.extend([label] * len(paths))
        logger.info(f"Subsampled image dataset to {len(image_paths)} images (max {max_samples} per class)")
    
    # Save label classes for deployment
    classes_path = output_dir / "label_classes.json"
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f)
        
    num_classes = len(class_names)
    
    # 2. Build Datasets
    batch_size = get_dynamic_batch_size(is_image_dataset=True)
    train_ds, val_ds, test_ds, val_labels, test_labels = build_image_datasets(
        image_paths, labels, class_names, batch_size=batch_size
    )
    
    # 3. Model construction
    # Attach data augmentation inside Keras model
    aug_pipe = get_augmentation_pipeline()
    model = ResNet9TF(
        input_shape=(224, 224, 3),
        num_classes=num_classes,
        augmentation_pipeline=aug_pipe
    )
    
    # Compile
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "sparse_categorical_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    # 4. Resume logic
    checkpoint_file = output_dir / "last_checkpoint.h5"
    state_file = output_dir / "training_state.json"
    initial_epoch = 0
    
    if checkpoint_file.exists() and state_file.exists():
        try:
            logger.info("Resuming training from last saved image checkpoint...")
            model = tf.keras.models.load_model(checkpoint_file, custom_objects={
                'GaussianNoiseLayer': GaussianNoiseLayer,
                'ColorAugmentationLayer': ColorAugmentationLayer
            })
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            initial_epoch = state.get("epoch", 0)
            logger.info(f"Resuming at epoch {initial_epoch + 1}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Building new model.")
            
    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(output_dir / "best_model.h5"), monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(TENSORBOARD_DIR / dataset_name), histogram_freq=1),
        TrainingStateCallback(output_dir)
    ]
    
    # 6. Train
    start_time = time.time()
    # Image models run for up to 300 epochs but EarlyStopping halts them at convergence
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    logger.info(f"Image training completed in {training_time:.2f} seconds.")
    
    # 7. Save curves
    plot_and_save_curves(history, output_dir, dataset_name)
    
    # 8. Load Best Model and Evaluate
    if (output_dir / "best_model.h5").exists():
        logger.info("Loading best model weights for evaluation...")
        model = tf.keras.models.load_model(output_dir / "best_model.h5")
        
    metrics_results = evaluate_classification_model(
        model, test_ds, test_labels, class_names, output_dir, dataset_name
    )
    
    # 9. Export formats
    export_model_formats(model, output_dir, "model", (224, 224, 3))
    
    # 10. Backwards Compatibility Copy
    handle_backwards_compatibility_copy(dataset_name, output_dir)
    
    # Clean up checkpoints
    try:
        if checkpoint_file.exists(): os.remove(checkpoint_file)
        if state_file.exists(): os.remove(state_file)
    except Exception:
        pass
        
    return metrics_results

def handle_backwards_compatibility_copy(dataset_name, output_dir):
    """
    Copies trained models to legacy directory paths and names expected by downstream components.
    """
    compat_dir = None
    target_name = None
    name_lower = dataset_name.lower()
    
    if name_lower == "fertilizer_prediction":
        compat_dir = MODEL_DIR / "fertilizer_recommendation"
        target_name = "fertilizer_recommender"
    elif name_lower == "crop_recommendation":
        compat_dir = MODEL_DIR / "crop_recommendation"
        target_name = "crop_recommender"
    elif name_lower == "price_prediction":
        compat_dir = MODEL_DIR / "price_prediction"
        target_name = "price_predictor"
    elif name_lower == "yield_prediction":
        compat_dir = MODEL_DIR / "yield_prediction"
        target_name = "yield_predictor"
    elif name_lower == "plant_disease":
        compat_dir = MODEL_DIR / "image_classification"
        target_name = "plant_resnet9"
    elif name_lower in ["cattle_diseases", "dog_skin_disease", "livestock"]:
        compat_dir = MODEL_DIR / "image_classification"
        target_name = f"animal_resnet9_{name_lower}" # Or map directly to animal_resnet9
        
    if compat_dir is not None:
        compat_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Copying model from {output_dir} to legacy path {compat_dir} with name prefix {target_name}...")
        
        # Copy the HDF5 model
        h5_src = output_dir / "model.h5"
        if h5_src.exists():
            shutil.copy2(h5_src, compat_dir / f"{target_name}.h5")
            
        # Copy the TFLite model
        tflite_src = output_dir / "model.tflite"
        if tflite_src.exists():
            shutil.copy2(tflite_src, compat_dir / f"{target_name}.tflite")
            
        # Copy the ONNX model
        onnx_src = output_dir / "model.onnx"
        if onnx_src.exists():
            shutil.copy2(onnx_src, compat_dir / f"{target_name}.onnx")
            
        # Copy metadata and pickles
        for f in output_dir.glob("*.json"):
            if f.name != "pipeline_summary.json":
                shutil.copy2(f, compat_dir / f.name)
        for f in output_dir.glob("*.pkl"):
            shutil.copy2(f, compat_dir / f.name)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Training Pipeline Orchestrator")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Override maximum training epochs")
    parser.add_argument("--dataset", type=str, default=None, help="Name of specific dataset directory to train")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum image samples per class (for faster runs)")
    args = parser.parse_args()

    set_seeds()
    device = detect_device()
    sys_info = get_system_usage()
    logger.info(f"System Configuration: CPU={sys_info['cpu_percent']}% | RAM={sys_info['ram_percent']}% ({sys_info['ram_used_gb']}/{sys_info['ram_total_gb']} GB)")
    
    # Detect all datasets in folder
    tabular_datasets, image_datasets = detect_datasets()
    
    # Filter datasets if targeted dataset name is provided
    if args.dataset:
        tabular_datasets = {k: v for k, v in tabular_datasets.items() if k.lower() == args.dataset.lower()}
        image_datasets = {k: v for k, v in image_datasets.items() if k.lower() == args.dataset.lower()}
        logger.info(f"Filtering datasets. Tabular={list(tabular_datasets.keys())}, Image={list(image_datasets.keys())}")
        
    summary = {}
    
    # 1. Train all detected Tabular models
    for dataset_name, csv_paths in tabular_datasets.items():
        try:
            metrics = train_tabular_model(dataset_name, csv_paths, max_epochs=args.epochs)
            summary[dataset_name] = {"status": "SUCCESS", "metrics": metrics}
        except Exception as e:
            logger.error(f"Error training tabular dataset {dataset_name}: {e}", exc_info=True)
            summary[dataset_name] = {"status": "FAILED", "error": str(e)}
            
    # 2. Train all detected Image models
    for dataset_name, data_dir in image_datasets.items():
        try:
            metrics = train_image_model(dataset_name, data_dir, max_epochs=args.epochs, max_samples=args.max_samples)
            summary[dataset_name] = {"status": "SUCCESS", "metrics": metrics}
        except Exception as e:
            logger.error(f"Error training image dataset {dataset_name}: {e}", exc_info=True)
            summary[dataset_name] = {"status": "FAILED", "error": str(e)}
            
    # Write final pipeline execution summary report
    summary_path = MODEL_DIR / "pipeline_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    logger.info("\n" + "="*70 + "\n PIPELINE SUMMARY REPORT\n" + "="*70)
    for dataset, info in summary.items():
        status = info["status"]
        if status == "SUCCESS":
            logger.info(f"[SUCCESS] {dataset:40} | STATUS: {status}")
            for k, v in info["metrics"].items():
                logger.info(f"   - {k:25}: {v:.4f}")
        else:
            logger.info(f"[FAILED]  {dataset:40} | STATUS: {status} | Error: {info['error']}")
    logger.info("="*70)
    logger.info(f"Pipeline complete. Summary written to {summary_path}")

if __name__ == "__main__":
    main()
