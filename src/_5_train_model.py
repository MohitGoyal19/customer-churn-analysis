import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

from src import config
from src.config import logger

# Define all the models and their parameters you want to train
models_to_train = {
	'RandomForest': {
		'model': RandomForestClassifier(random_state=42),
		'params': {'n_estimators': 100, 'max_depth': 10}
	},
	'LogisticRegression': {
		'model': LogisticRegression(random_state=42, max_iter=1000),
		'params': {'C': 1.0, 'solver': 'liblinear'}
	},
	'GradientBoosting': {
		'model': GradientBoostingClassifier(random_state=42),
		'params': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5}
	}
}

def train_model():
	'''Trains multiple models and logs experiments with MLflow.'''
	logger.info('Starting multi-model training task.')
	mlflow.set_experiment('Credit Card Churn Prediction')

	# Load feature data (this happens only once)
	df = pd.read_parquet(config.FEATURES_DIR / 'features.parquet')
	X = df.drop(columns=['target', 'event_timestamp'])
	y = df['target']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	# Start a parent MLflow run to group the models
	with mlflow.start_run(run_name='Model Comparison Run') as parent_run:
		logger.info(f'Parent MLflow run started with ID: {parent_run.info.run_id}')

		# Loop through each model defined in the dictionary
		for model_name, model_info in models_to_train.items():
			# Start a nested run for each model
			with mlflow.start_run(run_name=model_name, nested=True) as child_run:
				logger.info(f'--- Training {model_name} ---')

				# Instantiate and train the model
				model = model_info['model']
				model.set_params(**model_info['params'])
				model.fit(X_train, y_train)

				# Evaluate the model
				y_pred = model.predict(X_test)
				metrics = {
					'accuracy': accuracy_score(y_test, y_pred),
					'precision': precision_score(y_test, y_pred),
					'recall': recall_score(y_test, y_pred),
					'f1_score': f1_score(y_test, y_pred),
					'auc-roc_score': roc_auc_score(y_test, y_pred)
				}
				logger.info(f'Metrics for {model_name}: {metrics}')

				input_example = X_train.head(5)

				# Log everything to MLflow for this specific model
				mlflow.log_params(model_info['params'])
				mlflow.log_metrics(metrics)

				# Log the model artifact with a unique name
				mlflow.sklearn.log_model(
					model,
					registered_model_name=model_name,
					input_example=input_example
				)

	logger.info('Multi-model training task completed.')