import pandas as pd
from os.path import join
import settings

from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras


def show_score(name, y, prediction):
    print(name)
    print(confusion_matrix(y, prediction))
    print('accuracy\t:' + str(accuracy_score(y, prediction)))
    print('precision\t:' + str(precision_score(y, prediction)))
    print('recall\t:' + str(recall_score(y, prediction)))
    print('f1\t\t:' + str(f1_score(y, prediction)))


def train_data():
    x = pd.read_csv(join(settings.TRF_DATA_DIR, 'X.csv'))
    y = pd.read_csv(join(settings.TRF_DATA_DIR, 'y.csv'))

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=22)
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation=keras.activations.relu, input_shape=[9]),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(1, activation=keras.activations.sigmoid),
    ])
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.AUC()])

    model.fit(x_train, y_train, epochs=1000,
              validation_data=(x_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=20)])

    for name, x, y in (('Train set', x_train, y_train), ('Validation set', x_valid, y_valid)):
        prediction = (model.predict(x) > 0.5).astype('int32')
        show_score(name, y, prediction)

    model.save(join(settings.MODELS_DIR, 'ann_model.h5'))


def evaluate_model():
    x = pd.read_csv(join(settings.TRF_DATA_DIR, 'X.csv'))
    y = pd.read_csv(join(settings.TRF_DATA_DIR, 'y.csv'))

    model = keras.models.load_model(join(settings.MODELS_DIR, 'ann_model.h5'))

    prediction = (model.predict(x) > 0.5).astype('int32')
    show_score("Test set", y, prediction)
