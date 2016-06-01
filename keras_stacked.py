from stacking import StackModel

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.datasets import load_boston
import numpy as np
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from math import sqrt
from utils import tsv_batches_generator, read_tsv_batch
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD

from utils import tsv_batches_generator


class keras_online_model():
    def __init__(self, n_features, optimizer='rmsprop'):
        self.n_features = n_features
        self.optimizer = optimizer

        model = Sequential()
        model.add(Dense(400, batch_input_shape=(None, self.n_features)))
        model.add(PReLU())
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(200))
        model.add(PReLU())
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(100))
        model.add(PReLU())
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer=self.optimizer)

        self.model = model
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=1)

    def name(self):
        return "keras_defparams"

    def online_fit(self, paths_array):
        generator = tsv_batches_generator(paths_array, first_line=True, batchsize=1000)
        self.generator = generator
        self.model.fit_generator(generator.train(),
                            samples_per_epoch=2375000,
                            nb_epoch=100,
                            validation_data=generator.valid(),
                            nb_val_samples=1000,
                            callbacks=[self.early_stopping])

        valid = generator.valid().next()
        score = self.model.evaluate(valid[0], valid[1])
        print sqrt(score)


    def online_predict(self, inpath, outpath, first_line=False):
        print "predicting"
        print "writing to file"

        scaler = self.generator.standardscaler

        with open(outpath, 'a') as fout:
            for batch in read_tsv_batch(inpath, first_line=first_line, batchsize=1000):
                print batch
                x = batch[:, 1:]
                x[x == 'null'] = '0'
                x = x.astype(float)
                x = scaler.transform(x)
                pred = self.model.predict(x)
                for y in pred:
                    fout.write("{0:.3f}".format(y[0]) + '\n')

    def duplicate(self):
        return keras_online_model(self.n_features, optimizer=self.optimizer)

if __name__ == '__main__':
    base = "D:\\mfrik_data\\"
    uuid = "d915135f-c72a-499f-8697-e6c0834addf9"
    basepath = base + "stacking\\"

    # test_path is the testing dataset.
    test_path = base + "ccdm_test.tsv-preprocessed-ALL95.tsv"

    # Read header to know the number of features.
    with(open(test_path)) as fin:
        header = fin.readline().strip().split('\t')



    stackedmodel = StackModel(keras_online_model(len(header)-1), basepath, uuid, n_folds=5, verbose = True)
    stackedmodel.create_predictions(test_path)