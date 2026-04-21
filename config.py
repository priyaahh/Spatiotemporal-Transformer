import os

# Data Paths
DATA_FILE = "weatherHistory.csv"
MODEL_DIR = "saved_models"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset columns
FEATURES = ['Temp', 'Humidity', 'WindSpeed']
TARGET_COL = 'Temp'

# Sequence and Horizon Settings
SEQ_LENGTH = 10  # Number of past hours/days to use for prediction
DEFAULT_HORIZON = 7  # Default days to forecast into the future
SUPPORTED_HORIZONS = [7, 14, 30]

# General Training Parameters
BATCH_SIZE = 32
EPOCHS = 10  # Lightweight training

# LSTM Hyperparameters
LSTM_UNITS = 50

# Transformer Hyperparameters (Lightweight for CPU)
TRANSFORMER_HEADS = 2
TRANSFORMER_KEY_DIM = 32
TRANSFORMER_FF_DIM = 64
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_LAYERS = 1  # Kept small for CPU efficiency
