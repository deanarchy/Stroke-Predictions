import os

ROOT_DIR = os.path.abspath('')

DATA_DIR = os.path.join(ROOT_DIR, 'data')

TRF_DATA_DIR = os.path.join(DATA_DIR, 'transformed')

MODELS_DIR = os.path.join(ROOT_DIR, 'models')

DATASET_NAME = 'healthcare-dataset-stroke-data.csv'

TRAIN_DATASET_NAME = 'healthcare-dataset-stroke-data_train.csv'

TEST_DATASET_NAME = 'healthcare-dataset-stroke-data_test.csv'

KAGGLE_LINK = 'fedesoriano/stroke-prediction-dataset'
