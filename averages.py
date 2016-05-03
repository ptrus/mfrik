from sklearn_utils import GetTargetAverages
from utils import read_tsv, remove_outliers
from sklearn.utils import shuffle
from sklearn_utils import rmse_scorrer
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    avgs = ["ACCOUNTID", "CDNNAME", "SDK", "PLATFORM", "CREATIVETYPE", "EXTERNALADSERVER", "UA_DEVICETYPE"]

    #base="D:\\mfrik\\"
    base="/home/peterus/Projects/mfrik/"
    data,h = read_tsv(base + "ccdm_medium.tsv")
    print h
    data = remove_outliers(data, 0)
    print "outliers removed"
    y,x = data[:,0].astype(float), data[:, 1:]

    x,y = shuffle(x,y)

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


    # Test
    x = x[:, 0,5]
    tf0 = GetTargetAverages(0)
    tf5 = GetTargetAverages(1)
    ss = StandardScaler()
    reg = Ridge()


    p = Pipeline([('tf0', tf0), ('tf5', tf5), ('ss', ss), ('reg', reg)])
    params = dict(reg__alpha=[0.001, 0.1, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70])
    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_
