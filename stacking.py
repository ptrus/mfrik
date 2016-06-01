import numpy as np
import os
from cross_validation import get_full_train_paths, get_train_test_folds_paths
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn_utils import rmse_scorrer

class Stacker():
    def __init__(self, folds_preds, folds_target, predict_preds, stacking_model):
        self.folds_preds = folds_preds
        self.folds_target = folds_target
        self.predict_preds = predict_preds
        self.stacking_model = stacking_model

    def train(self):

        # Read level 0 folded predictions.
        X = []
        first = True
        for inpath in self.folds_preds:
            cntr = 0
            with open(inpath, 'r') as fin:
                for line in fin:
                    y = float(line.strip())
                    if first:
                        X.append([y])
                    else:
                        X[cntr].append(y)
                        cntr += 1
                first = False
        X = np.array(X)

        # Read level 0 folded targets.
        y = []
        with open(self.folds_target, 'r') as fin:
            for line in fin:
                y.append(float(line.strip()))
        y= np.array(y)

        # Fit level 1 model.
        self.stacking_model.fit(X, y)

    def predict(self):
        # Read level 0 predictions.
        X = []
        first = True
        for inpath in self.predict_preds:
            cntr = 0
            with open(inpath, 'r') as fin:
                for line in fin:
                    y = float(line.strip())
                    if first:
                        X.append([y])
                    else:
                        X[cntr].append(y)
                        cntr += 1
                first = False
        X = np.array(X)

        self.predict = self.stacking_model.predict(X)

    def write_results(self, out_path):
        with open(out_path, 'w') as fout:
            for y in self.predict:
                fout.write("{0:.3f}".format(y) + '\n')


class StackModel():
    def __init__(self, model, basepath, uuid, n_folds=5, verbose = True):
        self.model = model
        self.basepath = basepath
        self.uuid = uuid
        self.n_folds = n_folds
        self.verbose = verbose
        # Get outdir
        self.outdir = os.path.join(basepath, uuid + '-' + model.name() + '/')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def create_predictions(self, test_path):
        print "Starting CV predictions:"
        i = 0
        for trainpaths, testpath in get_train_test_folds_paths(self.basepath, self.uuid, n_folds = self.n_folds):
            i+=1
            if self.verbose: print "CV: %d / %d" % (i, self.n_folds)
            new_learner = self.model.duplicate()
            if self.verbose: print "Fitting..."
            new_learner.online_fit(trainpaths)
            if self.verbose: print "Predicting..."
            new_learner.online_predict(testpath, os.path.join(self.outdir, 'predict-folds.tsv')) # TODO: dont skip first line

        # Final fit
        print "Doing final fit."
        for testpath in get_full_train_paths(self.basepath, self.uuid, n_folds = self.n_folds):
            self.model.online_fit([testpath])

        self.model.online_predict(test_path, os.path.join(self.outdir, 'predict-final.tsv')) # TODO: skip first line

if __name__ == '__main__':
    base = "D:\\mfrik_data\\"
    folds_preds = [base + 'stacking\\d915135f-c72a-499f-8697-e6c0834addf9-keras_defparams\\predict-folds.tsv', base + 'stacking\\d915135f-c72a-499f-8697-e6c0834addf9-xgboost_defparams\\predict-folds.tsv']
    predict_preds = [base + 'stacking\\d915135f-c72a-499f-8697-e6c0834addf9-keras_defparams\\predict-final.tsv', base + 'stacking\\d915135f-c72a-499f-8697-e6c0834addf9-xgboost_defparams\\predict-final.tsv']

    folds_target = base + "stacking\\d915135f-c72a-499f-8697-e6c0834addf9-target"

    reg = Ridge()
    params = dict(alpha=[0.1])

    clf = GridSearchCV(reg, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=5)

    st = Stacker(folds_preds, folds_target, predict_preds, clf)

    st.train()
    st.predict()
    st.write_results(base + 'stacking\\stacked.tsv')