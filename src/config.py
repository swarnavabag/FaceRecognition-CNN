from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================
# DATA PATHS
# =========================
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_VGGFACE2_DIR = RAW_DATA_DIR / "VGGFace2"
PROCESSED_VGGFACE2_DIR = PROCESSED_DATA_DIR / "VGGFace2_160"

TRAIN_DIR = PROCESSED_VGGFACE2_DIR / "train"
VAL_DIR = PROCESSED_VGGFACE2_DIR / "val"

# =========================
# MODEL PATHS
# =========================
MODELS_DIR = PROJECT_ROOT / "models"

BEST_MODEL_PATH = MODELS_DIR / "face_recognition_inference.pth"
CHECKPOINT_PATH = MODELS_DIR / "face_recognition_checkpoint_epoch20.pth"

# =========================
# TRAINING CONFIG
# =========================
IMAGE_SIZE = 160
BATCH_SIZE = 32
EMBEDDING_DIM = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20