import utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy as sp
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn_utils import rmse_scorrer

if __name__ == "__main__":
    base="D:\\mfrik\\"
    #base="/home/peterus/Projects/mfrik/"
    data,h = utils.read_tsv(base+"outALL.tsv")

    x,y = data[:,1:],data[:,0]
    x[x=='null'] = '0'
    x = x.astype(float)
    y = y.astype(float)

    print h[1:], h[0]
    print x.shape, y.shape

    ss = StandardScaler()
    x = ss.fit_transform(x)

    sel = VarianceThreshold()
    x = sel.fit_transform(x)
    print x.shape

    x,y = shuffle(x,y)

    pca = PCA(0.8)
    x2 = pca.fit_transform(x)
    print x2.shape
    '''
    reg = linear_model.Ridge()
    parameters = {'alpha': [0.1, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70]}

    clf = GridSearchCV(reg, parameters, scoring=rmse_scorrer, n_jobs=1, verbose=True, cv=2)
    clf.fit(x, y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_
    '''

    # ONLINE LEARNinG
    x_test = x[-100:]
    y_test = y[-100:]
    x_train = x[:-100]
    y_train = y[:-100]
    sgd = SGDRegressor(alpha=1, penalty="l2")
    for i in range(3):
        batch = 1000
        for k in range(0, (len(x_train)/batch)-1):
            _x = x_train[batch*k:batch*k + batch]
            _y = y_train[batch*k:batch*k + batch]
            sgd.partial_fit(_x, _y)
        pred = rmse_scorrer(sgd, x_test, y_test)
        print pred
        #pred = sp.maximum(min, pred)
        #pred = sp.minimum(max, pred)
        #print "base:",mean_squared_error(y_test, np.repeat(sum(y_test)/len(y_test), len(y_test)))


    '''
    #quick plot
    #n, bins, patches = plt.hist(y, 50, normed=1, facecolor='green', alpha=0.75)
    #plt.show()
    '''
    #quick kmeans
    km = KMeans(n_clusters=4)
    km.fit(np.reshape(y, (len(y), 1)))
    print km.cluster_centers_
    #quick feature selection for outliers
    #y = km.labels_


    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import ExtraTreesRegressor
    forest = ExtraTreesRegressor(n_estimators=300,
                        n_jobs=-1)

    forest.fit(x, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %d (%f) - %s" % (f + 1, indices[f], importances[indices[f]], h[indices[f]+1]))

    clf = GridSearchCV(forest, {}, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_

    data,h = utils.read_tsv(base+"out_medium_timestamp.tsv")
    data = utils.remove_outliers(data, 0)
    print data.shape
    x,y = data[:,1:],data[:,0]
    x = x.astype(float)
    y = y.astype(float)
    print x.shape, y.shape

    # quick kmeans
    km = KMeans(n_clusters=4)
    km.fit(np.reshape(y, (len(y), 1)))
    print km.cluster_centers_

    from sklearn.dummy import DummyRegressor
    dummy = DummyRegressor()
    clf = GridSearchCV(dummy, {}, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x.reshape(-1,1), y)
    print "BASE:", clf.best_score_
    '''
    ns = [5, 10, 20, 30, 50, 60, 70, 90, 100, 110, 120, 140, 160, 180, 200, 300, 400]
    parameters = {'n_neighbors': ns, 'weights': ['uniform']}#, 'distance']}
    knn = KNeighborsRegressor()
    clf = GridSearchCV(knn, parameters, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x.reshape(-1,1), y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_
    '''
    reg = linear_model.Ridge()
    parameters = {'alpha': [0.001, 0.1, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70]}

    clf = GridSearchCV(reg, parameters, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_

    '''
    parameters = {'radius' : [10.0, 100.0, 200.0], 'weights': ['uniform']}#, 'distance']}
    rnn = RadiusNeighborsRegressor()
    clf = GridSearchCV(rnn, parameters, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x.reshape(-1,1), y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_
    '''
    '''
    ns = [1,2,3,4,5,10,20,30,50,60,70,90,100,110,120,140,160,180,200,300,400]
    parameters = {'n_neighbors': ns}
    knn = KNeighborsRegressor(metric="haversine")
    clf = GridSearchCV(knn, parameters, scoring=rmse_scorrer, n_jobs=-1, verbose=10, cv=2)
    print x[:,12:14].shape
    clf.fit(x[:,12:14], y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_
    '''
'''
Regression
sklearn.linear_model.SGDRegressor
sklearn.linear_model.PassiveAggressiveRegressor

Clustering
sklearn.cluster.MiniBatchKMeans

Decomposition / feature Extraction
sklearn.decomposition.MiniBatchDictionaryLearning
sklearn.decomposition.IncrementalPCA
sklearn.cluster.MiniBatchKMeans
'''