from flask import Blueprint, request, jsonify
from classification_model.predict import make_prediction
from classification_model import __version__ as _version

from api.config import get_logger
from api import __version__ as api_version
from api.validation import validate_inputs

import json
import numpy as np

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
	if request.method == 'GET':
	   _logger.info('health status OK')
	   return 'ok'
  
@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})

  
	   
@prediction_app.route('/v1/predict/classification', methods=['POST'])
def predict():
	if request.method == 'POST':
		#Extract POST data from request body as JSON
		json_data = request.get_json()
		json_data = json.dumps(json_data)
		_logger.info(f'Inputs: {json_data}')

		#Validate the input using marshmallow schema
		input_data, errors = validate_inputs(input_data=json_data)
		
		#Model prediction
		result = make_prediction(input_data=input_data)
		_logger.info(f'Outputs: {result}')

		#Get prediction and Convert numpy array to list
		predictions = result.get('predictions').tolist()
		version = result.get('version')
		
		
		#Return the response as JSON
		return jsonify({'predictions': predictions,
						'version': version,
						'errors': errors})