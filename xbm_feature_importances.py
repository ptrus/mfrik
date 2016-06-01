import xgboost as xgb
from scipy import sparse
import numpy as np

if __name__ == "__main__":
    base = "D:\\mfrik_data\\"
    base_output = "C:\\mfrik_data\\"
    # train = base+ file + "-preprocessed.tsv"
    train = base + 'all\\cdm_all.tsv-preprocessed.tsv'
    predict = base + 'all\\ccdm_test.tsv-preprocessed.tsv'

    params = {'alpha': 1,
              'colsample_bytree': 0.6,
              'silent': 0,
              'nthread': 7,
              'min_child_weight': 0.5,
              'subsample': 1,
              'eta': 0.1,
              'objective': 'reg:linear',
              'max_depth': 8,
              'gamma': 2,
              'lambda': 1}

    row = []
    col = []
    dat = []
    y = []
    with open(train, 'r') as fin:
        header = fin.readline().strip().split('\t')
        print(len(header))
        i = 0
        for line in fin:
            line = line.strip().split('\t')
            if i % 100000 == 0:
                print "at:", i
            for j, val in enumerate(line):
                if j == 0:
                    y.append(float(val))
                else:
                    if val != '0.0':
                        row.append(i)
                        col.append(j - 1)
                        dat.append(float(val))
            i += 1

    csr = sparse.csr_matrix((dat, (row, col)))
    dtrain = xgb.DMatrix(csr, label=y)
    gbdt = xgb.train(params.items(), dtrain, 400)

    importance = gbdt.get_fscore()
    import operator

    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    print importance
    names, vals = zip(*importance)
    print names, vals
    s = sum(vals)
    vals = [val / float(s) for val in vals]
    for i in range(50):
        print header[int(names[i][1:])+1], ":", vals[i]

    from collections import defaultdict
    importances_fixed = defaultdict(float)
    for key, val in importance:
        k = "".join(header[int(key[1:])+1].split("_")[:-1])
        if k == "":
            k = header[int(key[1:])+1]
        importances_fixed[k] += float(val)

    importances_fixed = sorted(importances_fixed.items(), key=operator.itemgetter(1), reverse=True)
    names, vals = zip(*importances_fixed)
    print names, vals
    s = sum(vals)
    vals = [val / float(s) for val in vals]
    for i in range(20):
        print names[i], ":", vals[i]
