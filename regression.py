import utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
import numpy as np

if __name__ == "__main__":
    data,h = utils.read_tsv("D:\\mfrik\\out.tsv")

    x,y = data[:,1:],data[:,0]
    x = x.astype(float)
    y = y.astype(float)
    print h[1:], h[0]
    print x.shape, y.shape

    ss = StandardScaler()
    x = ss.fit_transform(x)

    sel = VarianceThreshold()
    x = sel.fit_transform(x)
    print x.shape

    x,y = shuffle(x,y, random_state=42)

    pca = PCA(0.8)
    x2 = pca.fit_transform(x)
    print x2.shape

    reg = linear_model.Ridge()
    parameters = {'alpha': [0.1, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70]}

    clf = GridSearchCV(reg, parameters, scoring='mean_squared_error', n_jobs=-1, verbose=True, cv=5)
    clf.fit(x, y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_

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