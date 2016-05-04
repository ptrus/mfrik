import numpy as np
from utils import rmse, read_tsv_online
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from collections import defaultdict
from sklearn.neighbors import LSHForest, NearestNeighbors

def rmse_scorrer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    return rmse(y_true, y_pred)

def StandardScaler_online_fit(path, header=False):
    ss = StandardScaler()
    first = True
    for batch in read_tsv_online(path, maxmemusage=80):
        if first and header:
            first = False
            batch = batch[1:]
        ss.partial_fit(batch)
    return ss

def StandarScaler_online_transform(ss, inpath, outpath):
    with open(outpath, 'w') as fout:
        with open(inpath, 'r') as fin:
            for line in fin:
                if line and not line[0].isalpha(): # Skip header
                    l = np.array(line.strip().split('\t'))
                    l = ss.transform(l.astype(float).reshape(1, -1))

                    l = [str(ll) for ll in l[0]]
                    fout.write('\t'.join(l) + '\n')
                else:
                    fout.write(line)

def StandardScaler_inversetransform_col(ss, x, idx):
    ncomps = len(ss.scale_) # Find out number of features.
    v = np.zeros((len(x), ncomps))
    v[:,idx] = x
    v = ss.inverse_transform(v)
    return v[:, idx]


class GetTargetAverages(TransformerMixin):
    def __init__(self, idx):
        self.idx = idx
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def transform(self, X, *_):
        result = []
        for row in X:
            res = self.sums[row[self.idx]]
            cnt = self.counts[row[self.idx]]
            cnt = cnt if cnt > 0 else 1
            res = float(res) / cnt
            new_row = np.insert(row, self.idx, res)
            new_row = np.delete(new_row, self.idx+1)
            result.append(new_row)
        return np.array(result)

    def fit(self, X, y, *_):
        for i,row in enumerate(X):
            fval = row[self.idx]
            tval = float(y[i])
            self.sums[fval] += tval
            self.counts[fval] += 1
        return self

    def partial_fit(self, X, y, *_):
        self.fit(X, y, _)

    def get_params(self,deep=True):
        return { "idx" : self.idx }#,"sums" : self.sums,"counts" : self.counts }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class GetTargetClosest(TransformerMixin):
    def __init__(self, idx, num, n_estimators, n_candidates):
        self.idx = idx
        self.num = num
        self.n_estimators = n_estimators
        self.n_candidates = n_candidates
        self.lsh = LSHForest(n_candidates=n_candidates, n_estimators=n_estimators)
        self.nn = NearestNeighbors()

    def transform(self, X, *_):
        x = []
        for n in self.num:
            #knbh = self.lsh.kneighbors(X[:, self.idx].reshape(-1,1), n)
            dists, knbh = self.nn.kneighbors(X[:, self.idx].reshape(-1,1), n)
            x.append(knbh)
        x_new = []
        for xx in x:
            for row in xx:
                x_new.append(sum([float(self.target[i]) for i in row]))
        x_new = np.array(x_new).astype(float).reshape((len(self.num),X.shape[0])).T
        x_new = x_new / self.num
        #x = np.delete(x, self.idx, axis=1)
        #x = np.insert(x, x_new, axis=1)
        x = np.append(X, x_new, axis=1)
        return x

    def fit(self, X, y, *_):


        x = X[:, self.idx].reshape(-1,1)
        #self.lsh.fit(x)
        self.nn.fit(x)
        self.target = y
        self.x = x
        return self

    def partial_fit(self, X, *_):
        self.lsh.partial_fit(X, _)

    def get_params(self,deep=True):
        return { "idx" : self.idx, "num" : self.num, "n_estimators" : self.n_estimators, "n_candidates" : self.n_candidates }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
