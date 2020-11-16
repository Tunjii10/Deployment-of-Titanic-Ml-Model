import numpy as np
import pandas as pd

from classification_model.processing.data_management import load_pipeline
from classification_model.config import config
from classification_model.processing.validation import validate_inputs


pipeline_file_name = "classification_model.pkl"
_current_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    
	data = pd.read_json(input_data)
	
	validated_data = validate_inputs(input_data=data)

	prediction = _current_pipe.predict_test(validated_data)


	results = {"predictions": prediction}


	return results
