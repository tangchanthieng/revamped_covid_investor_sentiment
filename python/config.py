import pandas as pd
from pathlib import Path

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the path to the configuration file
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
MERGED_DATA_DIR = DATA_DIR / 'merged'
CLEANED_DATA_DIR = DATA_DIR / 'cleaned'

ASSETS_DIR = BASE_DIR / 'assets'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'