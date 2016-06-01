from datetime import datetime
from collections import Counter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.externals import joblib
from utils import read_tsv_batch
import pickle
import xgboost as xgb
from utils import read_tsv_online

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"
    base_output = "C:\\mfrik_data\\"
    #train = base+ file + "-preprocessed.tsv"
    train = base + 'cdm_all.tsv-preprocessed-with_cntriesALL.tsv'
    predict = base + 'ccdm_test.tsv-preprocessed-with_cntriesALL.tsv'

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
    data = []
    for batch in read_tsv_online(train):
        if data == []:
            data = batch[1:]
            data[data == 'null'] = '0'
            data = data.astype(float)
        break
        '''
        else:
            data2 = batch
            data2[data2 == 'null'] = '0'
            data2 = data2.astype(float)
            data = np.concatenate((data, data2), axis=0)
            break
        '''
    x,y = data[:,1:], data[:,0]
    print x.shape
    print "Fitting"

    dtest = xgb.DMatrix(x[-1000:], y[-1000:])
    dtrain = xgb.DMatrix(x[:-1000], y[:-1000])
    num_round = 10000
    evallist = [(dtrain, 'train'), (dtest, 'test')]
    bst = xgb.train(plst, dtrain, num_round, evallist, verbose_eval=100, early_stopping_rounds=100)
    models.append(bst)

    importance = bst.get_fscore()
    import operator
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    print importance
    names,vals = zip(*importance)
    s = sum(vals)
    vals = [val/float(s) for val in vals]
    for i in range(len(names)):
        print names[i], ":", vals[i]

    END()
    print 'Predicting'

    print "writing to file"
    with open(base + 'predictions_xb3.tsv', 'w') as fout:
        for batch in read_tsv_batch(predict, first_line=False, batchsize=10000):
            x = batch[:, 1:]
            x[x == 'null'] = '0'
            x = x.astype(float)
            pred = models[0].predict(xgb.DMatrix(x), ntree_limit=models[0].best_ntree_limit)
            for y in pred:
                fout.write("{0:.3f}".format(y) + '\n')