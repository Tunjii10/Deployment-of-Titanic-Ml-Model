import json

from classification_model.config import config
from classification_model.processing.data_management import load_dataset


def test_prediction_endpoint_validation_200(flask_test_client):

	test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
	post_json = test_data.to_json(orient='records')

	# When
	response = flask_test_client.post('/v1/predict/classification',
									  json=json.loads(post_json))

	# Then
	assert response.status_code == 200
	response_json = json.loads(response.data)
	# Check length of prediction equals input
	assert type(response_json)  == dict
	assert len(response_json.get('predictions')) == len(test_data)