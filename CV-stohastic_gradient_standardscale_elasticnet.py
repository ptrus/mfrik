from cross_validation import cv_predict
from sklearn.linear_model import SGDRegressor
from sklearn_utils import OnlineLearner

if __name__ == '__main__':
    # TODO: find an ok alpha in stohastic_gradient
    # TODO: add passive-aggressive regressor
    # TODO: add stohastics on different transformations
    alpha = 1
    online_learner = OnlineLearner(SGDRegressor(penalty='elasticnet', loss="squared_epsilon_insensitive", alpha=1), 10000, 'standardscale_elastic_svr', finditer=True)
    base_path = './data/binarized/'
    uuid= 'c951314c-03ea-4968-8590-86dbd377a7bf'
    prepreocessed = '__standardscale'
    n_folds = 5
    verbose = True

    cv_predict(online_learner, basepath=base_path, uuid=uuid, n_folds=n_folds, test_path='C:\\Users\\peteru\\Downloads\\ccdm_test.tsv-preprocessed.tsv_standardsacaled', prepreocessed=prepreocessed, verbose = verbose)

