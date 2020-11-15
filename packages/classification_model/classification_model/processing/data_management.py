import pandas as pd
import numpy as np
import joblib

from classification_model.config import config





def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
	"""Persist the pipeline"""

	# Prepare versioned save file name
	save_file_name = "classification_model.pkl"
	save_path = config.TRAINED_MODEL_DIR / save_file_name

	joblib.dump(pipeline_to_persist, save_path)

	print("saved pipeline")

def load_pipeline(*, file_name: str):
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


