from stacking import StackModel
from scipy import sparse
import xgboost as xgb

def_params = {'bst:max_depth': 8,
         'bst:eta': 0.01,
         'subsample': 0.8,
         'colsample_bytree': 0.7,
         'silent': 0,
         'objective': 'reg:linear',
         'nthread': 7
         }


class xgboost_online_model():
    def __init__(self, params = def_params, num_round = 300):
        self.params = params
        self.plist = self.params.items()
        self.num_round = num_round

    def name(self):
        return "xgboost_defparams"

    def online_fit(self, paths_array):
        # Online read sparse arrays.
        row = []
        col = []
        dat = []
        y = []

        print "Reading inputs"

        i = -1
        for path in paths_array:
            with open(path) as fin:
                for line in fin:
                    line = line.strip().split('\t')
                    for j, val in enumerate(line):
                        if j == 0:
                            y.append(float(val))
                            i += 1
                        else:
                            if val != '0.0':
                                row.append(i)
                                col.append(j - 1)
                                dat.append(float(val))

        csr = sparse.csr_matrix((dat, (row, col)))
        dtrain = xgb.DMatrix(csr, label=y)

        print "ended reading"

        print "start training"

        bst = xgb.train(self.plist, dtrain, self.num_round)

        print "stoped training"

        self.bst = bst


    def online_predict(self, inpath, outpath, first_line=False):
        print "predicting"
        print "writing to file"

        row_pred = []
        col_pred = []
        dat_pred = []
        with open(outpath, 'a') as fout:
            with open(inpath, 'r') as fin:
                if not first_line:
                    header = fin.readline().strip().split('\t')
                for i, line in enumerate(fin):
                    line = line.strip().split('\t')
                    if i % 100000 == 0:
                        print "at:", i
                    for j, val in enumerate(line):
                        if j == 0:
                            pass
                        else:
                            if val != '0.0':
                                row_pred.append(i)
                                col_pred.append(j - 1)
                                dat_pred.append(float(val))
                csr_pred = sparse.csr_matrix((dat_pred, (row_pred, col_pred)))
                pred = self.bst.predict(xgb.DMatrix(csr_pred))
                for y in pred:
                    fout.write("{0:.3f}".format(y) + '\n')

    def duplicate(self):
        return xgboost_online_model(self.params, num_round=self.num_round)

if __name__ == '__main__':
    base = "D:\\mfrik_data\\"
    uuid = "d915135f-c72a-499f-8697-e6c0834addf9"
    basepath = base + "stacking\\"

    # test_path is the testing dataset.
    test_path = base + "ccdm_test.tsv-preprocessed-ALL95.tsv"
    stackedmodel = StackModel(xgboost_online_model(), basepath, uuid, n_folds=5, verbose = True)
    stackedmodel.create_predictions(test_path)