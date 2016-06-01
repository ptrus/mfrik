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
from keras.optimizers import SGD


from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    base = "d:\\mfrik_data\\"
    #base = "C:\\Users\\peteru\\Downloads\\"

    file = "cdm_all.tsv"
    base_base = base + file
    without_outliers = base + file + "-without-outliers.tsv"
    shuffled_path = base + file + "-without-outliers-shuffled.tsv"
    preprocessed = base+ file + "-preprocessed-ALL95.tsv"

    out = []
    with open(preprocessed, 'r') as f:
        header = f.readline()
        header = header.strip().split('\t')
        t_idx = header.index('TIMESTAMP')

        for line in f:
            line = line.strip().split('\t')
            if datetime.fromtimestamp(float(str(line[t_idx]))).day == 4:
                out.append(line)

    out = np.array(out)
    out[out == 'null'] = '0'
    x, y = out[:, 1:].astype(float), out[:, 0].astype(float)
    print x.shape
    x = VarianceThreshold().fit_transform(x)
    x = StandardScaler().fit_transform(x)
    print x.shape
    X_test = x[-100:]
    y_test = y[-100:]
    X_train = x[:-100]
    y_train = y[:-100]

    print "done reading"


    model = Sequential()

    model.add(Dense(400, batch_input_shape=(None, x.shape[1])))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(200))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='linear'))
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='mse', optimizer=sgd)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.fit(X_train, y_train, nb_epoch=20, callbacks=[early_stopping])
    score = model.evaluate(X_test, y_test)
    print sqrt(score)
    print model.predict(X_test)
    print y_test

'''
model.train_on_batch(X, y)
model.test_on_batch(X, y)
model.fit_generator(data_generator, samples_per_epoch, nb_epoch).
'''