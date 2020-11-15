import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


class pipeline:
	
	
	def __init__(self, target,features, main_features,
					simple_imputer, categorical_encode):
		
		#initialize data_sets as none
		self.X_train = None
		self.Y_train = None
		self.X_test = None
		self.Y_test = None
		
		#variables to be engineered
		self.target = target
		self.features = features
		self.main_features = main_features
		self.simple_imputer = simple_imputer
		self.categorical_encode  = categorical_encode
	
		
		
		#model build
		self.classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth =5, random_state = 0)

		#feature engineering
		#i have two different imputers because different variables were missing in train and test set
		self.imputer1 = SimpleImputer(missing_values =np.nan, strategy='most_frequent')
		
		self.scalar = StandardScaler()

		self.categorical = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), self.categorical_encode)], remainder='passthrough')

		
	#***Create new features	****
	
	def title(self, df):
	#create new feature title
	
		df = df.copy()
		
		df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
		df['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
		df['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
		return df
	
	def family_type(self, df):
	#create new feature family_type
		df = df.copy()
				
		df['Fam_size'] = df['SibSp'] + df['Parch'] + 1

		df['Fam_Type'] = pd.cut(df['Fam_size'], [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

		return df
			
		
		
		
	#**Fit data for training**
	
	def fit(self, train_data):
		#pick x and y values
		self.X_train = train_data[self.features]
		self.Y_train = train_data[self.target]
		
		#create new features
		self.X_train = self.title(df = self.X_train)
		self.X_train = self.family_type(df = self.X_train)
		
		#drop features
		self.X_train = self.X_train[self.main_features]
		
		#impute missing values
		imp = self.X_train[self.simple_imputer].values.reshape(-1,1)
		self.imputer1.fit(imp)
		self.X_train[self.simple_imputer] = self.imputer1.transform(imp)
		
		#encode categorical values
		self.categorical.fit(self.X_train)
		self.X_train = np.array(self.categorical.transform(self.X_train))
		
		#separating dataset
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
        self.X_train, self.Y_train, test_size=0.1, random_state=0
		)
		
		#feature_scale
		fmp = self.X_train[:, [16]].reshape(-1,1)
		self.scalar.fit(fmp)
		self.X_train[:, [16]] = self.scalar.transform(fmp)
		self.X_test[:,[16]] = self.scalar.transform(self.X_test[:,[16]])
		
		#train model
		self.classifier.fit(self.X_train, self.Y_train)	
		
		return self
	
			
	#transform data for prediction
	def transform(self, data):
		#transform data for prediction
		data = data.copy()
		
		#create new feature
		data = self.title(df = data)
		data = self.family_type(df = data)
		
		#drop features
		data = data[self.main_features]
		
		#simple imputer
		imp = data[self.simple_imputer].values.reshape(-1,1)
		data[self.simple_imputer] = self.imputer1.transform(imp)
		

		#encode categorical values
		data = np.array(self.categorical.transform(data))
		
		#feature_scale
		fmp = data[:, [16]].reshape(-1,1)
		data[:, [16]] = self.scalar.transform(fmp)
		
		return data
		
	def predict_test(self, data):
	#predictions
		data = self.transform(data)
		
		prediction = self.classifier.predict(data)
		
		return prediction
	
	def evaluate_model(self):
	#evaluate model performance
		prediction_test = self.classifier.predict(self.X_test)
		print('Xtest accuracy: {}'.format((accuracy_score(self.Y_test, prediction_test))))
		print('confusion matrix: {}'.format((confusion_matrix(self.Y_test, prediction_test))))
		
		prediction_train = self.classifier.predict(self.X_train)
		print('Xtrain accuracy: {}'.format((accuracy_score(self.Y_train, prediction_train))))
		print('confusion matrix: {}'.format((confusion_matrix(self.Y_train, prediction_train))))
