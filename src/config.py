from dotenv import load_dotenv
import logging
import os
from pathlib import Path

# Load environment variables from .env file
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / '.env')

# ... (rest of the directory paths remain the same) ...
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / '01 Raw'
CLEAN_DATA_DIR = DATA_DIR / '02 Clean'
FEATURES_DIR = DATA_DIR / '03 Features'
REPORTING_DIR = DATA_DIR / '04 Reporting'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# --- DATA SOURCES ---
FILE_DATA_SOURCE = BASE_DIR / 'user_data/credit_info.csv'

# --- MongoDB SETTINGS ---
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DATABASE = os.getenv('MONGO_DATABASE')
MONGO_COLLECTION = os.getenv('MONGO_COLLECTION')

# --- Setup a logger ---
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'pipeline.log'),
        logging.StreamHandler() # To also print logs to the console
    ]
)

# Create a logger instance for the project
logger = logging.getLogger("ChurnPipeline")