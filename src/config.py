"""Central configuration for the potato leaf disease pipeline.

Every path can be overridden with an environment variable so the same code
runs locally, inside Docker and on a cloud platform such as Render.
"""
import os
from pathlib import Path

ROOT_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
TRAIN_DIR = Path(os.getenv("TRAIN_DIR", DATA_DIR / "train"))
TEST_DIR = Path(os.getenv("TEST_DIR", DATA_DIR / "test"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "potato_cnn.h5"))
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", MODELS_DIR / "class_names.json"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", MODELS_DIR / "metrics.json"))
INSIGHTS_PATH = Path(os.getenv("INSIGHTS_PATH", MODELS_DIR / "insights.json"))

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "potato_mlops.db"))

IMG_SIZE = int(os.getenv("IMG_SIZE", 128))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
SEED = 42

# Retraining defaults (fine-tuning the saved custom model)
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 5))
RETRAIN_LR = float(os.getenv("RETRAIN_LR", 1e-4))
# Minimum number of new uploaded images before retraining is allowed
MIN_NEW_IMAGES = int(os.getenv("MIN_NEW_IMAGES", 1))

# The Kaggle PlantVillage dataset also contains Pepper and Tomato folders. Only
# class folders starting with this prefix are used (if any match). "" = keep all.
CLASS_PREFIX = os.getenv("CLASS_PREFIX", "Potato")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for _d in (TRAIN_DIR, TEST_DIR, UPLOAD_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
