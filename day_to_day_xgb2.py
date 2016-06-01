from datetime import datetime
from collections import Counter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.externals import joblib
from utils import read_tsv_batch
import pickle
import xgboost as xgb

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"
    base_output = "C:\\mfrik_data\\"
    #train = base+ file + "-preprocessed.tsv"
    train = base + 'cdm_all.tsv-preprocessed-with_cntriesALL.tsv'
    predict = base + 'ccdm_test.tsv-preprocessed-with_cntriesALL.tsv'


    days = Counter()
    with open(train, 'r') as fin:
        header = fin.readline()
        header = header.strip().split('\t')
        t_idx = header.index('TIMESTAMP')
        for line in fin:
            line = line.strip().split('\t')
            date = datetime.fromtimestamp(float(str(line[t_idx])))
            days[(str(date.month) + '_' + str(date.day))] += 1

    xm1 = [(key,val) for key,val in days.items() if val > 10000]
    xm2 = [(key,val) for key,val in days.items() if val <= 10000]
    print xm2, sum((val) for _, val in xm2)

    param = {'bst:max_depth': 6,
             'bst:eta': 0.01,
             'subsample': 0.8,
             'colsample_bytree': 0.7,
             'silent': 0,
             'objective': 'reg:linear',
             'nthread': 7
             }

    plst = param.items()
    num_round = 20000
    models = []
    for key,_ in xm1:
        print "At:", key
        [month, day] = key.split('_')
        data = []
        print "Reading input"
        with open(train) as fin:
            header = fin.readline()
            header = header.strip().split('\t')
            print header
            t_idx = header.index('TIMESTAMP')
            for line in fin:
                line = line.strip().split('\t')
                date = datetime.fromtimestamp(float(str(line[t_idx])))
                if str(date.month) == month and str(date.day) == day:
                    data.append(line)
        data = np.array(data)
        data[data == 'null'] = '0'
        data = data.astype(float)
        x,y = data[:,1:], data[:,0]
        print x.shape
        print "Fitting"

        dtest = xgb.DMatrix(x[-100:], y[-100:])
        dtrain = xgb.DMatrix(x[:-100], y[:-100])
        num_round = 10000
        evallist = [(dtrain, 'train'), (dtest, 'test')]
        bst = xgb.train(plst, dtrain, num_round, evallist, verbose_eval=100, early_stopping_rounds=100)
        models.append(bst)

    print 'Predicting'

    print "writing to file"
    with open(base + 'day_to_day_predictions_xb2.tsv', 'w') as fout:
        with open(predict, 'r') as fin:
            fin.readline()
            for row in fin:
                x = np.array(row)
                x[x == 'null'] = '0'
                x = x[1:]
                x = x.astype(float)
                x = xgb.DMatrix(x.reshape(1, -1))
                pred = 0
                for model in models:
                    pred += model.predict(x, ntree_limit=model.best_ntree_limit)
                pred = pred / len(models)
                fout.write( "{0:.3f}".format(pred) + '\n')