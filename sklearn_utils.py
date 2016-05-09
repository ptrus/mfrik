import numpy as np
from utils import rmse, read_tsv_online, read_tsv_batch, read_tsvs_batch
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from collections import defaultdict
from sklearn.neighbors import LSHForest, NearestNeighbors
from sklearn.externals import joblib
from cross_validation import get_train_test_folds_paths
import copy
import sys

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

class OnlineLearner():
    def __init__(self, learner, batchsize, name, targetidx=0, n_iters=1, finditer=False, *_):
        self.batchsize = batchsize
        self.learner = learner
        self.name = name
        self.targetidx = targetidx
        self.n_iters = n_iters
        self.finditer = finditer

    def online_fit(self, inpaths):

        if not self.finditer:
            for i in range(self.n_iters):
                for batch in read_tsvs_batch(inpaths, batchsize=self.batchsize):
                    if len(batch) == 0: continue
                    x = np.delete(batch, self.targetidx, axis=1).astype(float)
                    y = batch[:, self.targetidx].astype(float)
                    self.learner.partial_fit(x, y)
        else:
            n_iters = 0
            print "Searching nr. of iterations"
            for i in range(10000):
                print "At iter: %d" % (i)
                first = True
                for batch in read_tsvs_batch(inpaths, batchsize=self.batchsize):
                    if first:
                        ''' First batch will be used for testing convergence '''
                        test_x = np.delete(batch, self.targetidx, axis=1).astype(float)
                        test_y = batch[:, self.targetidx].astype(float)
                        first = False
                        continue

                    if len(batch) == 0: continue
                    y = batch[:, self.targetidx].astype(float)
                    x = np.delete(batch, self.targetidx, axis=1).astype(float)
                    self.learner.partial_fit(x, y)

                pred = self.learner.predict(test_x)
                print pred
                score = -rmse(test_y, pred)
                avg2 = np.average(test_y)
                score2 = -rmse(test_y, avg2)
                print "Score: %.2f" % (score)
                print "Score avg: %.2f" % (avg2)

                if i == 0:
                    old_score = score
                    continue
                ''' Stop if worse RMSE score '''
                if score <= old_score: old_score = score
                else:
                    n_iters = i
                    break

            print "Stopped after iters: %d" % (n_iters)
            self.n_iters = n_iters
            ''' Final fit on all '''
            print "Fitting..."
            for i in range(n_iters):
                print "At iter %d/%d" % (i, n_iters)
                for batch in read_tsvs_batch(inpaths, batchsize=self.batchsize):
                    if len(batch) == 0: continue
                    x = np.delete(batch, self.targetidx, axis=1).astype(float)
                    y = batch[:, self.targetidx].astype(float)
                    self.learner.partial_fit(x, y)
        return self

    def online_transform(self, inpath, outpath, transform_target = False):
        with open(outpath, 'a') as fout:
            for batch in read_tsv_batch(inpath, batchsize=self.batchsize):
                if len(batch) == 0: continue
                if transform_target:
                    x_new = self.learner.transform(batch).astype(float)
                else:
                    x = np.delete(batch, self.targetidx, axis=1).astype(float)
                    y = batch[:, self.targetidx]
                    y[y == 'null'] = 0
                    x_new = self.learner.transform(x)
                    x_new = np.insert(x_new, self.targetidx, y, axis=1)
                for row in x_new:
                    row = [str(x) for x in row]
                    fout.write('\t'.join(row) + '\n')

    def online_predict(self, inpath, outpath):
        with open(outpath, 'a') as fout:
            for batch in read_tsv_batch(inpath, batchsize=self.batchsize, first_line=True):
                if len(batch) == 0: continue
                x = np.delete(batch, self.targetidx, axis=1).astype(float)
                x_new = self.learner.predict(x)
                for row in x_new:
                    row = "{0:.3f}".format(row)#print("%.2f" % a)

                    fout.write(row + '\n')

    def save(self, path):
        joblib.dump(self, path)

    def load(self, path):
        return joblib.load(path)

    def duplicate(self):
        return copy.copy(self)