from kaggle.api.kaggle_api_extended import KaggleApi
import settings
import os


def fetch_data():
    if not os.path.exists(settings.DATA_DIR):
        os.mkdir(settings.DATA_DIR)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(settings.KAGGLE_LINK, settings.DATASET_NAME, path=settings.DATA_DIR)
