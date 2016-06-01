from utils import read_tsv_batch, read_tsv, rmse, read_tsv_online, rmse
from sklearn.linear_model import SGDRegressor
from sklearn_utils import rmse_scorrer, StandardScaler_online_fit, StandarScaler_online_transform, StandardScaler_inversetransform_col, OnlineLearner
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib

def SGD_fit(train_path, n_iters, batchsize=1000, loss='squared_loss', penalty='l2', alpha=1, l1_ratio=0.15, postprocess=None):
    sgd = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio)
    old_score = None
    iters = 0
    while iters < n_iters:
        iters += 1
        bnum = 0
        for batch in read_tsv_batch(train_path, first_line=True, batchsize=batchsize):
            bnum += 1
            if bnum % 100 == 0:
                print "At batch: %d" % bnum
            x, y = batch[:, 1:], batch[:, 0]
            x[x == 'null'] = '0'
            x = x.astype(float)
            y = y.astype(float)
            if postprocess is not None:
                y = StandardScaler_inversetransform_col(ss, y, 0)
            sgd.partial_fit(x,y)

        if conv_test_y != []:# and iters % 10 == 0:
            pred = sgd.predict(conv_test_x)
            print "Predicted:"
            print pred
            print "TEST:"
            print conv_test_y
            #if postprocess is not None:
            #    pred = postprocess.inverse_transform(pred)

            print "In iteration: %d" % (iters)
            if old_score is None:
                old_score = rmse(conv_test_y, pred)
                continue
            score = rmse(conv_test_y, pred)
            print "Old score: %2.f, new score: %2.f" % (old_score, score)
            if score < old_score:
                break
            old_score = score
            joblib.dump(sgd, "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\" + 'sgd-alpha_' + str(alpha) + 'iters_' + str(iters) + 'score_' + str(old_score) + '.pkl')
    print "FInished after %d iters." % (iters)

    return sgd

if __name__ == '__main__':
    #base = "D:\\mfrik\\"
    #base="/home/peterus/Projects/mfrik/"
    #base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "C:\\Users\\peteru\\mfrik\\data\\binarized\\"

    ss = StandardScaler_online_fit(base + 'ccdm_large_preprocessed.tsv')
    StandarScaler_online_transform(ss, base + 'ccdm_large_preprocessed.tsv', base + 'ccdm_large_preprocessed-scaled.tsv')
    StandarScaler_online_transform(ss, base + 'final_train.tsv', base + 'final_train-scaled.tsv')
    joblib.dump(ss, base + 'ssonline.pkl')
    '''
    ss = joblib.load(base + 'ssonline.pkl')
    test,h = np.array(read_tsv(base+'final_test-scaled.tsv', header=True))
    testX,testY = test[:, 1:], test[:, 0]
    testX = testX.astype(float)
    testY = testY.astype(float)
    testYT = StandardScaler_inversetransform_col(ss, testY.astype(float), 0)
    mean = np.mean(testYT)
    print "Base score %2.f" % (rmse(testYT, mean))

    alphas = 10.0**-np.arange(-1,7)
    for alpha in alphas:
        sgd = SGD_fit(base + "final_train-scaled.tsv", 10000, testX, testYT, 10000, alpha=alpha, postprocess=ss)
        #joblib.dump(sgd, base + 'sgd-fitted.pkl')
        #sgd = joblib.load(base + 'sgd-fitted.pkl')
    '''