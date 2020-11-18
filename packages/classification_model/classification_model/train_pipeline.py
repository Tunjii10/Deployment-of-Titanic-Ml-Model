import numpy as np
import pandas as pd
import joblib

from classification_model import pipeline
from classification_model.config import config

from classification_model.processing.data_management import load_dataset, save_pipeline
from classification_model import __version__ as _version

import logging

_logger = logging.getLogger('classification_model')


	
def run_training() -> None:
	
	# read training data
	data = load_dataset(file_name=config.TRAINING_DATA_FILE)	
    
	
	#actual training by calling pipeline class
	pipeline.pipeline.fit(data)

    #save pipeline
	_logger.info(f"saving model version: {_version}")
	save_pipeline(pipeline_to_persist=pipeline.pipeline)


if __name__ == "__main__":
    run_training()
	