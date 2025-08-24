import pandas as pd

from src import config
from src.config import logger

def validate_data(date_str: str):
	'''Performs data quality checks on the raw ingested files.'''
	logger.info('Starting data validation task.')
	validation_success = True

	# --- Validate Credit CSV Data ---
	try:
		path = config.RAW_DATA_DIR / f'credit_info/{date_str}/credit_info.csv'
		df = pd.read_csv(path)

		if 'CLIENTNUM' not in df.columns or df['CLIENTNUM'].isnull().any():
			logger.error('Validation FAIL: \'CLIENTNUM\' in CSV is missing or has nulls.')
			validation_success = False

	except Exception as e:
		logger.error(f'Validation FAIL: Could not read or validate credit_info.csv. Error: {e}')
		validation_success = False

	# --- Validate Personal Info JSON Data ---
	try:
		path = config.RAW_DATA_DIR / f'customer/{date_str}/customer.json'
		df = pd.read_json(path, lines=True)

		if '_id' not in df.columns or df['_id'].isnull().any():
			logger.error('Validation FAIL: \'_id\' in MongoDB data is missing or has nulls.')
			validation_success = False

	except Exception as e:
		logger.error(f'Validation FAIL: Could not read or validate customer.json. Error: {e}')
		validation_success = False

	if not validation_success:
		raise ValueError('Data validation failed. Halting pipeline.')

	logger.info('Data validation task completed successfully.')

validate_data('2025-08-24')