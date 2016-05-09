from cross_validation import cv_predict
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn_utils import OnlineLearner

if __name__ == '__main__':
    #base = "D:\\mfrik\\"
    #base="/home/peterus/Projects/mfrik/"
    base = "C:\\Users\\peteru\\Downloads\\"
    train_file = base + 'ccdm_large_preprocessed.tsv'
    test_file = base + 'ccdm_test.tsv-preprocessed.tsv'
    '''
    print "Sandard scaling..."
    onlinescaler = OnlineLearner(StandardScaler(), 10000, 'standardscaler')
    print "Fitting..."
    onlinescaler.online_fit([train_file])
    print "Transforming..."
    onlinescaler.online_transform(train_file, str(train_file) + '_standarscaled')
    onlinescaler.online_transform(test_file, str(test_file) + '_standardsacaled')
    #onlinescaler.save(base +'__standardscaler_fitted_all')
    '''
    alphas = [5,2,7,3,6,4]
    for alpha in alphas:
        print 'Training alphas'
        online_learner = OnlineLearner(SGDRegressor(penalty='elasticnet', loss="squared_epsilon_insensitive", alpha=alpha), 10000, 'standardscaler', finditer=True)
        online_learner.online_fit([str(train_file) + '_standarscaled'])
        online_learner.online_predict(str(test_file) + '_standardsacaled', str(test_file) + '_standardsacaled_elasticnet_svr'+'_alpha_' + str(alpha))
