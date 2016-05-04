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
    print StandardScaler()
    print StandardScaler().partial_fit
    onlinescaler = OnlineLearner(StandardScaler(), 10000, 'standardscaler')
    for train in get_full_train_paths(base_path=base_path, uuid=uuid, prepreocessed='', n_folds=n_folds):
        temp_learner = onlinescaler.duplicate()
        temp_learner.online_fit(train)
        temp_learner.online_transform(train, train+out_prepreocessed)

    for train in get_full_train_paths():
        onlinescaler.online_fit(train)

    onlinescaler.save(os.join(base_path, uuid+'__standardscaler_fitted_all'))
