import os
import json
import time
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
import psutil
import tf2onnx

from ai.config import (
    MODEL_DIR, TENSORBOARD_DIR, RANDOM_SEED, MAX_EPOCHS,
    DEFAULT_BATCH_SIZE, HP_TUNING_RUNS, OPTIMIZERS
)
from ai.utils import logger, detect_device, get_system_usage, plot_and_save_curves
from ai.models.resnet9_tf import ResNet9TF
from ai.augmentation import get_augmentation_pipeline

class TrainingStateCallback(tf.keras.callbacks.Callback):
    """Callback to persist current epoch and state to resume if interrupted."""
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        
    def on_epoch_end(self, epoch, logs=None):
        state = {
            "epoch": epoch + 1,
            "logs": {k: float(v) for k, v in logs.items()} if logs else {}
        }
        with open(self.output_dir / "training_state.json", "w") as f:
            json.dump(state, f)
        # Save last model checkpoint
        self.model.save(self.output_dir / "last_checkpoint.h5")

def get_dynamic_batch_size(is_image_dataset=False):
    """Dynamically determine the best batch size based on available memory and GPU."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if is_image_dataset:
            # Scale batch size for images on GPU
            return 32 if available_gb > 8 else 16
        else:
            return 64
    else:
        # Scale batch size on CPU
        return 16 if is_image_dataset else 32

def get_optimizer(name, lr=1e-3):
    """Factory function for Keras optimizers."""
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        logger.warning(f"Unknown optimizer {name}, defaulting to Adam")
        return tf.keras.optimizers.Adam(learning_rate=lr)

def build_tabular_model(input_dim, output_dim, task_type, hp=None):
    """Constructs a feed-forward MLP for tabular regression/classification."""
    if hp is None:
        hp = {
            "hidden_layers": [64, 32],
            "dropout": 0.2,
            "l2_reg": 1e-4,
            "activation": "relu"
        }
        
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    
    for units in hp["hidden_layers"]:
        x = layers.Dense(
            units,
            kernel_regularizer=tf.keras.regularizers.l2(hp["l2_reg"]),
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(hp["activation"])(x)
        x = layers.Dropout(hp["dropout"])(x)
        
    if task_type == "regression":
        outputs = layers.Dense(output_dim)(x)
    elif task_type == "binary_classification":
        outputs = layers.Dense(output_dim, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(output_dim, activation="softmax")(x)
        
    model = Model(inputs, outputs, name="tabular_mlp")
    return model

def select_best_optimizer(build_model_fn, train_data, val_data, epochs=3):
    """Runs a mini-experiment to see which optimizer yields the best performance."""
    logger.info("Starting optimizer comparison experiment...")
    best_opt = "adam"
    best_val_loss = float('inf')
    
    # Unpack training inputs/labels
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    for opt_name in OPTIMIZERS:
        model = build_model_fn()
        opt = get_optimizer(opt_name, lr=1e-3)
        
        # Binary or Multiclass or Regression
        if model.layers[-1].activation.__name__ == 'softmax':
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        elif model.layers[-1].activation.__name__ == 'sigmoid':
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
            
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        
        # Train for a few epochs
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        final_val_loss = history.history['val_loss'][-1]
        logger.info(f"Optimizer {opt_name} - Val Loss: {final_val_loss:.4f}")
        
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_opt = opt_name
            
    logger.info(f"Optimizer {best_opt} selected as the best.")
    return best_opt

def export_model_formats(model, output_dir, name, input_shape):
    """Exports model to SavedModel, HDF5 (.h5), TFLite, and ONNX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. HDF5 Format
    h5_path = output_dir / f"{name}.h5"
    model.save(h5_path)
    logger.info(f"Exported HDF5 model to {h5_path}")
    
    # 2. SavedModel Format
    sm_path = output_dir / "saved_model"
    try:
        model.export(sm_path)
        logger.info(f"Exported SavedModel to {sm_path}")
    except Exception as e:
        logger.warning(f"Could not use model.export, attempting fallback: {e}")
        try:
            model.save(sm_path)
            logger.info(f"Saved model to {sm_path}")
        except Exception as e2:
            logger.error(f"Failed to export SavedModel format: {e2}")
    
    # 3. TensorFlow Lite Format
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Enable Select TF ops for compatibility with complex layers
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        tflite_path = output_dir / f"{name}.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        logger.info(f"Exported TFLite model to {tflite_path}")
    except Exception as e:
        logger.error(f"Failed to export TFLite model: {e}")
        
    # 4. ONNX Format
    try:
        onnx_path = output_dir / f"{name}.onnx"
        spec = (tf.TensorSpec((None,) + input_shape, model.inputs[0].dtype, name="input"),)
        tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(onnx_path))
        logger.info(f"Exported ONNX model to {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")

def tune_hyperparameters(build_model_fn, train_data, val_data, task_type, num_features):
    """Tunes learning rate, dropout, and hidden layers via Random Search."""
    logger.info("Starting hyperparameter tuning...")
    best_val_loss = float('inf')
    best_hp = None
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Search parameters
    lrs = [1e-2, 1e-3, 5e-4]
    dropouts = [0.1, 0.2, 0.3]
    hidden_configs = [
        [128, 64],
        [64, 32],
        [32, 16]
    ]
    
    for i in range(HP_TUNING_RUNS):
        lr = np.random.choice(lrs)
        dropout = np.random.choice(dropouts)
        hidden = hidden_configs[np.random.choice(len(hidden_configs))]
        
        hp = {
            "hidden_layers": hidden,
            "dropout": dropout,
            "l2_reg": 1e-4,
            "activation": "relu"
        }
        
        model = build_model_fn(hp)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
        if task_type == "regression":
            loss = "mse"
        elif task_type == "binary_classification":
            loss = "binary_crossentropy"
        else:
            loss = "sparse_categorical_crossentropy"
            
        model.compile(optimizer=opt, loss=loss)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,
            batch_size=32,
            verbose=0
        )
        
        val_loss = history.history['val_loss'][-1]
        logger.info(f"HP Run {i+1}: lr={lr}, dropout={dropout}, hidden={hidden} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hp = hp
            best_hp["lr"] = lr
            
    logger.info(f"Hyperparameter tuning completed. Best configuration: {best_hp}")
    return best_hp
