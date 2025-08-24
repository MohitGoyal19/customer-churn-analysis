import pandas as pd
import re

from src import config
from src.config import logger


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	'''Converts all column names to snake_case.'''
	new_cols = {}
	for col in df.columns:
		if col == 'CLIENTNUM':
			new_cols[col] = 'client_no'
		
		else:
			# Replace capital letters with underscore + lowercase
			s = re.sub(r'(?<!^)(?=[A-Z])', '_', col)
			# 2. Replace any characters that are not letters/numbers with an underscore,
			#    and convert the entire string to lowercase.
			new_cols[col] = re.sub(r'[^a-zA-Z0-9]+', '_', s).lower()

	df = df.rename(columns=new_cols)

	logger.info('Standardized all column names to snake_case.')

	return df


def prepare_data(date_str: str):
	'''Joins data, standardizes column names, and cleans the final dataset.'''
	logger.info('Starting data preparation task.')

	# Load raw data
	df_account = pd.read_csv(config.RAW_DATA_DIR / f'credit_info/{date_str}/credit_info.csv')
	df_customer = pd.read_json(config.RAW_DATA_DIR / f'customer/{date_str}/customer.json', lines=True)

	# Rename MongoDB join key to match CSV key
	df_customer.rename(columns={'_id': 'CLIENTNUM'}, inplace=True)

	# Merge datasets
	df = pd.merge(df_customer, df_account, on='CLIENTNUM', how='inner')
	logger.info(f'Successfully merged datasets. Shape of merged data: {df.shape}')

	# Standardize all column names to snake_case
	df = standardize_columns(df)

	# use 'attrition_flag'. We will rename it to 'target' for clarity.
	if 'attrition_flag' in df.columns:
		df.rename(columns={'attrition_flag': 'target'}, inplace=True)
		# Convert target to binary 0/1
		df['target'] = df['target'].apply(lambda x: 1 if 'Existing Customer' not in x else 0)
		logger.info('Processed target variable \'attrition_flag\'.')

	else:
		logger.warning('Target column \'attrition_flag\' not found. Model training will fail.')

	# Save cleaned data
	config.CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
	clean_data_path = config.CLEAN_DATA_DIR / f'clean_data_{date_str}.csv'

	df.to_csv(clean_data_path, index=False)
	logger.info(f'Cleaned and standardized data saved to {clean_data_path}')
	logger.info('Data preparation task completed.')