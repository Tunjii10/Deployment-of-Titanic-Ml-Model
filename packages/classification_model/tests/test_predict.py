import math

from classification_model.predict import make_prediction
from classification_model.processing.data_management import load_dataset


def test_make_single_prediction():
	# Given
	test_data = load_dataset(file_name='test.csv')
	single_test_json = test_data[0:1].to_json(orient='records')

	# When
	subject = make_prediction(input_data=single_test_json)

	# Then
	assert subject is not None
	assert math.ceil(subject.get('predictions')[0]) == 0

