import numpy as np
from utils import rmse, read_tsv_online
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

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
    def __init__(self, idxs):
        self.idxs = idxs
        self.averages = {}

    def transform(self, X, y, *_):
        result = []
        for index, rowdata in X.iterrows():
            rowdict = {}
            for kvp in self.kpairs:
                rowdict.update( { rowdata[ kvp[0] ]: rowdata[ kvp[1] ] } )
            result.append(rowdict)
        return result

    def fit(self, *_):
        return self
