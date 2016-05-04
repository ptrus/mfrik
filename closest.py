from sklearn_utils import GetTargetClosest
from utils import read_tsv, remove_outliers
from sklearn.utils import shuffle
from sklearn_utils import rmse_scorrer
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.neighbors import LSHForest, NearestNeighbors

if __name__ == '__main__':
    #base="D:\\mfrik\\"
    #base="/home/peterus/Projects/mfrik/"
    base="C:\\Users\\peteru\\mfrik\\"
    data,h = read_tsv(base + "ccdm_medium.tsv")
    print h
    data = remove_outliers(data, 0)
    print "outliers removed"
    y,x = data[:,0].astype(float), data[:, 1:]

    x = x[:, 4]
    x,y = shuffle(x,y)

    x[x=='null'] = '0'

    x = np.array(x).astype(float).reshape(-1,1)

    h = h[1:]
    h = h[4]
    #lsh = LSHForest(n_estimators=10, n_candidates=5)
    #nn = NearestNeighbors()
    #print "nh"
    #nn.fit(x)
    #print "fitted"
    #print nn.kneighbors(x, 10)

    #idx =
    nums = [1,2,3,4,5,10,20,100]
    h = np.append(h, ["ts-"+str(n) for n in nums])
    ct1 = GetTargetClosest(idx=0, num=nums, n_candidates=50, n_estimators=100)

    ss = StandardScaler()
    reg = Ridge()
    #reg = ExtraTreesRegressor(n_estimators=500, n_jobs=-1)
    ss = StandardScaler()
    p = Pipeline([('ct1', ct1),('ss', ss),('reg', reg)])

    params = dict(reg__alpha=[0.0001,0.001, 0.1, 1, 5, 10, 50, 100, 1000])
    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_
    print clf.best_estimator_


    importances = clf.best_estimator_.steps[-1][1].feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(len(h)):
        print("%d. feature %d (%f) - %s" % (f, indices[f], importances[indices[f]], h[indices[f]]))



