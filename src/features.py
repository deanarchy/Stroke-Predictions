import pandas as pd
import os
import joblib
import settings

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.utils import resample


def split_data():
    data_location = os.path.join(settings.DATA_DIR, settings.DATASET_NAME)
    x_train, x_test = train_test_split(pd.read_csv(data_location), random_state=22)
    x_train.to_csv(path_or_buf=data_location[:-4] + '_train.csv', index=False)
    x_test.to_csv(path_or_buf=data_location[:-4] + '_test.csv', index=False)


def transform_data(train=False, test=False):
    if train:
        data_location = os.path.join(settings.DATA_DIR, settings.TRAIN_DATASET_NAME)
    if test:
        data_location = os.path.join(settings.DATA_DIR, settings.TEST_DATASET_NAME)

    raw_data = pd.read_csv(data_location)

    if train:
        # Resampling
        false_majority = raw_data[raw_data['stroke'] == 0].copy()
        true_minority = raw_data[raw_data['stroke'] == 1].copy()

        true_minority_resampled = resample(true_minority,
                                           replace=True,
                                           n_samples=len(false_majority),
                                           random_state=22)

        assert len(false_majority) == len(true_minority_resampled)
        raw_data = pd.concat([true_minority_resampled, false_majority])

    stroke_features = raw_data.drop(['id', 'bmi', 'stroke'], axis=1)
    stroke_labels = raw_data['stroke'].copy()

    numerical = stroke_features.select_dtypes(exclude='O')
    categorical = stroke_features.select_dtypes(include='O')

    if train:
        pipeline = ColumnTransformer([
            ('scaler', StandardScaler(), list(numerical)),
            ('encoder', OrdinalEncoder(), list(categorical)),
        ])

        stroke_features = pipeline.fit_transform(stroke_features)
        joblib.dump(pipeline, os.path.join(settings.MODELS_DIR, 'pipeline.pkl'))

    if test:
        pipeline = joblib.load(os.path.join(settings.MODELS_DIR, 'pipeline.pkl'))
        stroke_features = pipeline.transform(stroke_features)

    if not os.path.exists(settings.TRF_DATA_DIR):
        os.mkdir(settings.TRF_DATA_DIR)
    pd.DataFrame(stroke_features).to_csv(path_or_buf=os.path.join(settings.TRF_DATA_DIR, 'X.csv'), index=False)
    pd.DataFrame(stroke_labels).to_csv(path_or_buf=os.path.join(settings.TRF_DATA_DIR, 'y.csv'), index=False)
