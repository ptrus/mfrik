from datetime import datetime
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from sklearn_utils import rmse_scorrer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from operator import itemgetter
from utils import rmse
from sklearn.feature_selection import VarianceThreshold

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    base = "d:\\mfrik_data\\"
    #base = "C:\\Users\\peteru\\Downloads\\"

    file = "cdm_all.tsv"
    base_base = base + file
    without_outliers = base + file + "-without-outliers.tsv"
    shuffled_path = base + file + "-without-outliers-shuffled.tsv"
    preprocessed = base+ file + "-preprocessed-with_cntriesALL.tsv"

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
    x,y = out[:,1:].astype(float), out[:,0].astype(float)
    vt = VarianceThreshold()
    base = sum(y) / len(y)
    base = rmse(y, [base]*len(y))
    print "Baseline score: ", base

    forest = ExtraTreesRegressor(n_estimators=300,
                                 n_jobs=-1)

    forest.fit(x, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %d (%f) - %s" % (f + 1, indices[f], importances[indices[f]], header[indices[f] + 1]))

    ids = [indices[:10], indices[:20], indices[:40], indices[:50], indices[:80], indices[:100], indices[:200]]

    for idx in ids:
        print "Taking indices ",  len(idx)
        for id in idx:
            print header[id + 1]

        xtemp = x[:,idx]
        print xtemp.shape

        print "Shape before:", xtemp.shape
        xtemp = vt.fit_transform(xtemp)
        print "Shape after:", xtemp.shape
        xtemp, ytemp = shuffle(xtemp, y)


        et = ExtraTreesRegressor(n_jobs=-1)
        params = dict(n_estimators = [500])
        clf = GridSearchCV(et, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
        clf.fit(xtemp, ytemp)
        print "ExtraTreesRegressor:"
        scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
        for score in scores:
            print score

    end()


    reg = Ridge()
    params = dict(reg__alpha= [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000])
    ss = StandardScaler()

    p = Pipeline([('ss', ss), ('reg', reg)])
    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print "Ridge normal:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score


    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x2, y)
    print "Ridge sqrt:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score

    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x3, y)
    print "Ridge log:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score
    '''
    svr = SVR(C=1)
    p = Pipeline([('s', s), ('svr', svr)])
    params = dict(svr__C= [0.01, 0.1, 1, 10, 100], svr__kernel=['rbf', 'linear'])
    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print "Svr normal:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score


    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x2, y)
    print "Svr sqrt:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score


    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x3, y)
    print "Svr log:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score
    '''
