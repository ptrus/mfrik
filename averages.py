from sklearn_utils import GetTargetAverages
from utils import read_tsv, remove_outliers
from sklearn.utils import shuffle
from sklearn_utils import rmse_scorrer
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

if __name__ == '__main__':
    avgs = ["ACCOUNTID", "CDNNAME", "SDK", "PLATFORM", "CREATIVETYPE", "EXTERNALADSERVER", "UA_DEVICETYPE"]

    #base="D:\\mfrik\\"
    #base="/home/peterus/Projects/mfrik/"
    base="C:\\Users\\peteru\\mfrik\\"
    data,h = read_tsv(base + "ccdm_medium.tsv")
    print h
    data = remove_outliers(data, 0)
    print "outliers removed"
    y,x = data[:,0].astype(float), data[:, 1:]

    x,y = shuffle(x,y)

    x[x=='null'] = '0'

    h = h[1:]

    '''
    idxs = [h.index(a) for a in avgs]
    x = x[:, idxs]
    print x.shape
    print h[idxs]
    get averages for indexes
    '''

    #tf = GetTargetAverages(0)
    #tf.fit(x, y)
    #print tf.sums
    #print tf.counts

    '''
    x = tf.transform(x)
    print x
    print x[:, -1]
    '''


# geoip lat, lng ... timestamp

    # Test
    used = [0, 14, 21, 5, 22, 7, 33, 43, 39]#, 4, 23, 24]
    x = x[:, used]
    h = np.array(h)[used]
    y = y.astype(float)
    print x.shape
    print h
    print x[1]
    tf0 = GetTargetAverages(0)
    tf14 = GetTargetAverages(1)
    tf21 = GetTargetAverages(2)
    tf5 = GetTargetAverages(3)
    tf22 = GetTargetAverages(4)
    tf7 = GetTargetAverages(5)
    tf33 = GetTargetAverages(6)
    tf43 = GetTargetAverages(7)
    tf39 = GetTargetAverages(8)
    ss = StandardScaler()
    reg = Ridge()
    #reg = ExtraTreesRegressor(n_estimators=500, n_jobs=-1)
    p = Pipeline([('tf0', tf0), ('tf14', tf14),('tf21', tf21),('tf5', tf5),('tf22', tf22),('tf7', tf7),('tf33', tf33), ('tf43', tf43),('tf39', tf39), ('ss', ss),('reg', reg)])
    #p = Pipeline([('tf0', tf0), ('tf22', tf22), ('tf7', tf7), ('ss', ss),('reg', reg)])

    params = dict(reg__alpha=[0.0001,0.001, 0.1, 1, 5, 10, 50, 100, 1000])
    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_
    print clf.best_estimator_

    '''
    importances = clf.best_estimator_.steps[-1][1].feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %d (%f) - %s" % (f, indices[f], importances[indices[f]], h[indices[f]]))
    '''

    '''
0. feature 0 (0.269039) - ACCOUNTID
1. feature 4 (0.179928) - CREATIVETYPE
2. feature 5 (0.120460) - EXTERNALADSERVER
3. feature 1 (0.116660) - SDK
4. feature 8 (0.102931) - UA_OS
5. feature 6 (0.070946) - UA_DEVICETYPE
6. feature 3 (0.065929) - PLATFORM
7. feature 2 (0.052422) - CDNNAME
8. feature 7 (0.021685) - UA_BROWSERRENDERINGENGINE
    '''