from pathlib import Path
import os

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the path to the configuration file
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
STAGING_DATA_DIR = DATA_DIR / 'staging'
MART_DATA_DIR = DATA_DIR / 'mart'

ASSETS_DIR = BASE_DIR / 'assets'
NOTEBOOKS_DIR = BASE_DIR / 'notebook'
