from scipy import sparse
import xgboost as xgb
from utils import read_tsv_batch
import numpy as np
from sklearn.grid_search import RandomizedSearchCV
from sklearn_utils import rmse_scorrer

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"
    base_output = "C:\\mfrik_data\\"
    # train = base+ file + "-preprocessed.tsv"
    train = base + 'cdm_all.tsv-preprocessed.tsv'
    predict = base + 'ccdm_test.tsv-preprocessed.tsv'

    param = {'max_depth': 6,
             'eta': 0.3,
             'subsample': 0.8,
             'colsample_bytree': 0.7,
             'silent': 0,
             'objective': 'reg:linear',
             'nthread': 7
             }

    plst = param.items()
    num_round = 20000

    row = []
    col = []
    dat = []
    y = []
    y_test = []
    row_test = []
    col_test = []
    dat_test = []
    with open(train, 'r') as fin:
        header = fin.readline().strip().split('\t')
        i = 0
        i_test = 0
        for line in fin:
            line = line.strip().split('\t')
            if i % 100000 == 0:
                print "at:",i
            if i < 300000:
                for j, val in enumerate(line):
                    if j == 0:
                        y.append(float(val))
                    else:
                        if val != '0.0':
                            row.append(i)
                            col.append(j-1)
                            dat.append(float(val))
                i += 1
            else:
                for j, val in enumerate(line):
                    if j == 0:
                        y_test.append(float(val))
                    else:
                        if val != '0.0':
                            row_test.append(i_test)
                            col_test.append(j - 1)
                            dat_test.append(float(val))
                i_test += 1
            if i_test  == 300000:
                break

    csr = sparse.csr_matrix((dat, (row, col)))
    csr_test = sparse.csr_matrix((dat_test, (row_test, col_test)))
    dtrain = xgb.DMatrix(csr, label=y)
    dtest = xgb.DMatrix(csr_test, label=y_test)

    print "ended reading"

    print "start training"
    evallist = [(dtrain, 'train'), (dtest, 'test')]

    bst = xgb.train(plst, dtrain, num_round, evallist, verbose_eval=100, early_stopping_rounds=200)

    print "stoped training"

    stopping_limit = bst.best_ntree_limit
    # Tune:
    # max_depth, min_child_weight, gamma, subsample, colsample_bytree

    reg = xgb.XGBRegressor(n_estimators=bst.best_ntree_limit, learning_rate=0.1)
    params = dict(max_depth=[3,6,8,10,12,14], min_child_weight=[0,0.5,1,2,5], gamma=[0,0.5,1,2,5], subsample=[0.5,0.6,0.8,1], colsample_bytree=[0.5,0.6,0.8,1])

    clf = RandomizedSearchCV(reg, params, scoring=rmse_scorrer, n_jobs=1, n_iter=20, verbose=1, cv=2)
    clf.fit(csr,y)

    # Tune:
    # lambda, alpha
    params = clf.best_params_
    print("Best params:", params)
    reg = xgb.XGBRegressor()
    reg.set_params(**params)

    params2 = {"reg_lambda":[1e-5, 1e-2, 0.1, 1,100], "reg_alpha":[1e-5, 1e-2, 0.1, 1,100]}
    clf = RandomizedSearchCV(reg, params2, scoring=rmse_scorrer, n_jobs=1, n_iter=10, verbose=1, cv=2)
    # Tune:
    clf.fit(csr, y)

    # lower learning rate tune number of trees
    params2 = clf.best_params_

    param = {'max_depth': params['max_depth'],
             'min_child_weight': params['min_child_weight'],
             'gamma': params['gamma'],
             'subsample':params['subsample'],
             'colsample_bytree': params['colsample_bytree'],
             'alpha': params2['reg_alpha'],
             'lambda': params2['reg_lambda'],
             'silent': 0,
             'objective': 'reg:linear',
             'nthread': 7,
             'eta': 0.01,
             }

    print("Best params:", param)
    end()
    row = []
    col = []
    dat = []
    y = []
    y_test = []
    row_test = []
    col_test = []
    dat_test = []
    with open(train, 'r') as fin:
        header = fin.readline().strip().split('\t')
        i = 0
        i_test = 0
        for line in fin:
            line = line.strip().split('\t')
            if i % 100000 == 0:
                print "at:", i
            if i < 2340000:
                for j, val in enumerate(line):
                    if j == 0:
                        y.append(float(val))
                    else:
                        if val != '0.0':
                            row.append(i)
                            col.append(j - 1)
                            dat.append(float(val))
                i += 1
            else:
                for j, val in enumerate(line):
                    if j == 0:
                        y_test.append(float(val))
                    else:
                        if val != '0.0':
                            row_test.append(i_test)
                            col_test.append(j - 1)
                            dat_test.append(float(val))
                i_test += 1

    csr = sparse.csr_matrix((dat, (row, col)))
    csr_test = sparse.csr_matrix((dat_test, (row_test, col_test)))
    dtrain = xgb.DMatrix(csr, label=y)
    dtest = xgb.DMatrix(csr_test, label=y_test)
    evallist = [(dtrain, 'train'), (dtest, 'test')]
    plist = param.items()
    bst = xgb.train(plist, dtrain, num_round, evallist, verbose_eval=100, early_stopping_rounds=200)

    print "predicting"
    print "writing to file"

    row_pred = []
    col_pred = []
    dat_pred = []
    with open(base + 'predictions_xb_sparse.tsv', 'w') as fout:
        with open(predict, 'r') as fin:
            header = fin.readline().strip().split('\t')
            for i, line in enumerate(fin):
                line = line.strip().split('\t')
                if i % 100000 == 0:
                    print "at:", i
                for j, val in enumerate(line):
                    if j == 0:
                        pass
                    else:
                        if val != '0.0':
                            row_pred.append(i)
                            col_pred.append(j - 1)
                            dat_pred.append(float(val))
            csr_pred = sparse.csr_matrix((dat_pred, (row_pred, col_pred)))
            pred = bst.predict(xgb.DMatrix(csr_pred), ntree_limit=bst.best_ntree_limit)
            for y in pred:
                fout.write("{0:.3f}".format(y) + '\n')

    print "end"
