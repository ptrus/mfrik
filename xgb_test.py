import os
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
import scipy

if __name__ == '__main__':

    (x,y) = make_regression(n_samples=100, n_features=20,n_informative=3)

    param = {'bst:max_depth': 6,
             'bst:eta': 0.01,
             'subsample': 0.8,
             'colsample_bytree': 0.7,
             'silent': 1,
             'objective': 'reg:linear',
             'nthread': 7
             }
    plst = param.items()

    row = []
    col = []
    dat = []
    i = 0
    for i,line in enumerate(x):
        for j,val in enumerate(line):
            row.append(i)
            col.append(j);
            dat.append(float(val))

    csr = scipy.sparse.csr_matrix((dat, (row, col)))
    dtrain = xgb.DMatrix(csr, label=y)
    bst = xgb.cv(plst, dtrain, 100)
    print bst

    dtrain = xgb.DMatrix(x, label=y)
    bst = xgb.cv(plst, dtrain, 100)
    print bst
    end()
    dtest = xgb.DMatrix(x[-2:], y[-2:])
    dtrain = xgb.DMatrix(x[:-2], y[:-2])
    num_round = 1000
    plst = param.items()

    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }

    evallist = [(dtrain, 'train'), (dtest, 'test')]
    bst = xgb.train(plst, dtrain, num_round, evallist, verbose_eval=50, early_stopping_rounds=50)
    print bst.best_ntree_limit
    results = xgb.cv(plst, dtrain, 1000, 2, early_stopping_rounds=10)
    n_trees = results.shape[0]
    gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(learning_rate=0.01, n_estimators=n_trees, max_depth=6,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='reg:linear', nthread=4, scale_pos_weight=1),
                            param_grid=param_test1, n_jobs=4, iid=False, cv=5)
    gsearch1.fit(x,y)
    print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    bst.save_model('test.bin')
    joblib.dump(bst, 'test.bin')
    bst = xgb.Booster({'nthread': 4})  # init model
    bst = joblib.load("test.bin")  # load data
    print bst.best_ntree_limit
    end()
    importance = bst.get_fscore()
    import operator
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    print importance
    names,vals = zip(*importance)
    s = sum(vals)
    vals = [val/float(s) for val in vals]
    for i in range(len(names)):
        print names[i], ":", vals[i]

    xgb.plot_importance(bst)


    #print vars(bst)
    print bst.get_fscore()
    print bst.predict(xgb.DMatrix(x))
    print y

    bst.save_model('test.bin')
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model("test.bin")  # load data
    print bst.predict(xgb.DMatrix(x))

