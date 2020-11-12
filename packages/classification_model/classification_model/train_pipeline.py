import numpy as np
import pandas as pd
import joblib

from classification_model import pipeline
from classification_model.config import config

def save_pipeline(*,pipeline_to_persist) -> None:
	
	
	save_file_name = 'classification_model.pkl'
	save_path = config.TRAINED_MODEL_DIR / save_file_name
	joblib.dump(pipeline_to_persist, save_path)
	
	print("saved pipeline")
	
def run_training() -> None:
	
	data = pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE)
	
   
	pipeline.pipeline.fit_transform(data)

	print("saving pipeline")
	save_pipeline(pipeline_to_persist=pipeline.pipeline)


if __name__ == "__main__":
    run_training()
	