import os
from pathlib import Path

# Paths
AI_DIR = Path(__file__).resolve().parent
REPO_ROOT = AI_DIR.parent

DATASET_DIR = REPO_ROOT / "datasets"
MODEL_DIR = AI_DIR / "models"
LOG_DIR = AI_DIR / "logs"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Training Configurations
MAX_EPOCHS = 300
DEFAULT_BATCH_SIZE = 32

# Splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Image dataset settings
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3

# Hyperparameter search config
HP_TUNING_RUNS = 5

# Supported Optimizers
OPTIMIZERS = ["adam", "adamw", "rmsprop", "sgd"]
