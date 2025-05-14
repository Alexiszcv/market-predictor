from pathlib import Path

# Racine du projet
ROOT_DIR = Path(__file__).resolve().parents[2]


DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FULL_DATASET = PROCESSED_DIR/"dataset_full.parquet"

PRED_DIR = DATA_DIR / "predictions"



