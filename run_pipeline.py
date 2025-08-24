import subprocess
import sys

from src.config import logger

# Import pipeline step functions
from src._1_ingest import ingest_data
from src._2_validate import validate_data
from src._3_prepare import prepare_data
from src._4_build_features import build_features
from src._5_train_model import train_model


def main():
	'''Runs the end-to-end ML pipeline.'''
	logger.info(' --- Starting End-to-End Churn Prediction Pipeline ---')

	try:
		date_str = ingest_data()
		validate_data(date_str)
		prepare_data(date_str)
		build_features(date_str)

		logger.info('Versioning data with DVC...')
		subprocess.run(
			[sys.executable, "-m", "dvc", "add", "data/03 Features/features.parquet"],
			check=True
		)

		logger.info('Data versioned.')

		train_model()

		logger.info('--- Pipeline Execution Finished Successfully! ---')

	except Exception as e:
		logger.error(f'--- Pipeline Failed: {e} ---', exc_info=True)


if __name__ == '__main__':
	main()