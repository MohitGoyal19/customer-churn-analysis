from datetime import datetime
import os
import pandas as pd
from pymongo import MongoClient

from src import config
from src.config import logger


def ingest_data():
	'''Reads data from CSV and MongoDB and saves to the raw data layer.'''
	logger.info('Starting data ingestion task.')
	date_str = datetime.now().strftime('%Y-%m-%d')

	# Ingest from CSV
	try:
		logger.info(f'Reading customer data from {config.FILE_DATA_SOURCE}')
		customer_output_dir = config.RAW_DATA_DIR / f'credit_info/{date_str}'
		os.makedirs(customer_output_dir, exist_ok=True)

		df_customer = pd.read_csv(config.FILE_DATA_SOURCE)
		df_customer.to_csv(customer_output_dir / 'credit_info.csv', index=False)
		logger.info(f'Successfully ingested {len(df_customer)} customer records from CSV.')

	except FileNotFoundError:
		logger.error(f'CSV file not found at {config.FILE_DATA_SOURCE}. Please check the path.')
		raise

	except Exception as e:
		logger.error(f'Error ingesting from CSV: {e}')
		raise

	# Ingest from MongoDB
	try:
		logger.info(f'Connecting to MongoDB at {config.MONGO_URI}...')
		customer_output_dir = config.RAW_DATA_DIR / f'customer/{date_str}'
		os.makedirs(customer_output_dir, exist_ok=True)

		client = MongoClient(config.MONGO_URI)
		db = client[config.MONGO_DATABASE]
		collection = db[config.MONGO_COLLECTION]

		cursor = collection.find({})
		df_customer = pd.DataFrame(list(cursor))
		client.close()

		if df_customer.empty:
			logger.warning('No documents found in the MongoDB collection.')

		df_customer.to_json(customer_output_dir / 'customer.json', orient='records', lines=True)
		logger.info(f'Successfully ingested {len(df_customer)} service records from MongoDB.')

	except Exception as e:
		logger.error(f'Error ingesting from MongoDB: {e}')
		raise

	logger.info('Data ingestion task completed.')

	return date_str