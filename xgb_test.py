import xgboost as xgb
from sklearn.datasets import make_regression

if __name__ == '__main__':

    (x,y) = make_regression(n_samples=50, n_features=4,n_informative=1)

    param = {'bst:max_depth': 2,
             'bst:eta': 0.01,
             'subsample': 0.8,
             'colsample_bytree': 0.7,
             'silent': 0,
             'objective': 'reg:linear',
             'nthread': 7
             }

    dtrain = xgb.DMatrix(x, y)
    num_round = 1000
    plst = param.items()

    bst = xgb.train(plst, dtrain, num_round)
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

