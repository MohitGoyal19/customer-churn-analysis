from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.config import logger

def build_features(date_str: str):
	'''Engineers features from the clean, standardized dataset.'''
	logger.info('Building features task started.')

	clean_data_path = config.CLEAN_DATA_DIR / f'clean_data_{date_str}.csv'
	df = pd.read_csv(clean_data_path)

	# Drop the original client identifier as it's not a feature
	if 'clientnum' in df.columns:
		df = df.drop(columns=['clientnum'])

	# One-Hot Encode categorical variables
	categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
	df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
	logger.info(f'Performed one-hot encoding on columns: {categorical_cols}')

	# Scale numerical features
	numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
	if 'target' in numerical_cols:
		numerical_cols.remove('target')

	scaler = StandardScaler()
	df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
	logger.info(f'Performed scaling on {len(numerical_cols)} numerical features.')

	# Add timestamp for Feast feature store
	df['event_timestamp'] = pd.to_datetime(datetime.now())

	# Save feature data
	config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
	feature_data_path = config.FEATURES_DIR / 'features.parquet'
	df.to_parquet(feature_data_path, index=False)

	logger.info(f'Feature dataset built and saved to {feature_data_path}')