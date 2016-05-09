from cross_validation import get_full_train_paths
from sklearn.preprocessing import StandardScaler
from sklearn_utils import OnlineLearner
import os

if __name__ == '__main__':
    base_path = './data/binarized/'
    uuid= 'c951314c-03ea-4968-8590-86dbd377a7bf'
    prepreocessed = ''
    out_prepreocessed='__standardscale'
    n_folds = 5
    onlinescaler = OnlineLearner(StandardScaler(), 10000, 'standardscaler')
    i=0

    for train in get_full_train_paths(base_path=base_path, uuid=uuid, prepreocessed='', n_folds=n_folds):
        i += 1
        print "Fold %d" % (i)
        temp_learner = onlinescaler.duplicate()
        print "Fitting ..."
        temp_learner.online_fit([train])
        print "Transforming..."
        print "From: ", train
        print "To: ", (str(train)+out_prepreocessed)
        temp_learner.online_transform(train, str(train)+out_prepreocessed)

    for train in get_full_train_paths(base_path=base_path, uuid=uuid, prepreocessed='', n_folds=n_folds):
        onlinescaler.online_fit([train])

    onlinescaler.save(os.path.join(base_path, uuid+'__standardscaler_fitted_all'))
