import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

from src import config
from src.config import logger

def train_model():
	'''Trains a model and logs experiments with MLflow.'''
	logger.info('Starting model training task.')

	# Set up MLflow experiment
	mlflow.set_experiment('Credit Card Churn Prediction')

	with mlflow.start_run():
		# Load feature data
		df = pd.read_parquet(config.FEATURES_DIR / 'features.parquet')

		# Define features (X) and target (y)
		X = df.drop(columns=['target', 'event_timestamp'])
		y = df['target']

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
		logger.info(f'Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).')

		# --- Model Training ---
		params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
		model = RandomForestClassifier(**params)
		model.fit(X_train, y_train)

		# --- Evaluation ---
		y_pred = model.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		precision = precision_score(y_test, y_pred)
		recall = recall_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred)

		metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
		logger.info(f'Model evaluation metrics: {metrics}')

		# --- MLflow Logging ---
		mlflow.log_params(params)
		mlflow.log_metrics(metrics)
		mlflow.sklearn.log_model(model, 'random_forest_model')

		logger.info('Model training complete and logged to MLflow.')

train_model()