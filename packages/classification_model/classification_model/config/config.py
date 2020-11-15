import pathlib

import classification_model

import pandas as pd

PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent

TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

DATASET_DIR = PACKAGE_ROOT / "datasets"

TRAINING_DATA_FILE = "train.csv"

TESTING_DATA_FILE= "test.csv"

TARGET = "Survived"


FEATURES = ['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Fare',
			'Embarked']

MAIN_FEATURES =  ['Pclass', 'Sex', 'Fare', 'Embarked', 'Title',
				'Fam_Type']				
				
SIMPLE_IMPUTER = 'Embarked'
				
				
CATEGORICAL_ENCODE = ['Sex', 'Embarked','Title',
				'Fam_Type']
				
