from utils import read_tsv_batch, read_tsv, rmse, read_tsv_online, rmse
from sklearn.linear_model import SGDRegressor
from sklearn_utils import rmse_scorrer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib

def SGD_fit(train_path, n_iters, conv_test_x=[], conv_test_y=[], batchsize=1000, loss='squared_loss', penalty='l2', alpha=30, l1_ratio=0.15, postprocess=None):
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
            sgd.partial_fit(x,y)

        if conv_test_y != []:# and iters % 10 == 0:
            joblib.dump(sgd, "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\" + 'sgd-fitted.pkl')
            pred = sgd.predict(conv_test_x)
            #if postprocess is not None:
            #    pred = postprocess.inverse_transform(pred)

            print "In iteration: %d" % (iters)
            if old_score is None:
                old_score = rmse(conv_test_y, pred)
                continue
            score = rmse(conv_test_y, pred)
            print "Old score: %2.f, new score: %2.f" % (old_score, score)
            if score < old_score:
                pass
                #break
            old_score = score
    print "FInished after %d iters." % (iters)

    return sgd

class StandardScalerOnline():
    def __init__(self):
        self.means = None
        self.maxs = None
        self.sums = None
        self.len = None

    def fit(self, inpath):
        for batch in read_tsv_online(inpath):
            if batch[0][0][0].isalpha(): # Header
                batch = batch[1:]
            batch = batch.astype(float)
            if self.sums is None:
                self.sums = np.sum(batch, 0)
                self.len = len(batch)
                self.maxs = np.max(np.abs(batch), 0)
            else:
                self.sums += np.sum(batch, 0)
                self.len += len(batch)
                self.maxs = np.maximum(self.maxs, np.max(np.abs(batch), 0))
        self.means = self.sums / self.len
        self.maxs = self.maxs - self.means
        self.maxs[self.maxs == 0] = 1

    def transform(self, inpath, outpath):
        with open(outpath, 'w') as fout:
            with open(inpath, 'r') as fin:
                for line in fin:
                    if line and not line[0].isalpha():
                        l = np.array(line.strip().split('\t'))
                        oldl = np.array(l)
                        l = l.astype(float)
                        l = (l - self.means) / self.maxs
                        l = [str(ll) for ll in l]
                        fout.write('\t'.join(l) + '\n')
                    else:
                        fout.write(line)

    def inverse_transform(self, vals, idx=0):
        return (vals * self.maxs[idx]) + self.means[idx]

if __name__ == '__main__':
    #base = "D:\\mfrik\\"
    #base="/home/peterus/Projects/mfrik/"
    base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"

    '''
    ss = StandardScalerOnline()
    ss.fit(base + 'final_train.tsv')
    ss.transform(base + 'final_test.tsv', base + 'final_test-scaled.tsv')
    ss.transform(base + 'final_train.tsv', base + 'final_train-scaled.tsv')
    joblib.dump(ss, base + 'ssonline.pkl')
    '''
    ss = joblib.load(base + 'ssonline.pkl')

    test,h = np.array(read_tsv(base+'final_test-scaled.tsv', header=True))
    testX,testY = test[:, 1:], test[:, 0]
    testX = testX.astype(float)
    testY = testY.astype(float)
    testYT = ss.inverse_transform(testY.astype(float))
    mean = np.mean(testYT)
    print "Base score %2.f" % (rmse(testYT, mean))

    sgd = SGD_fit(base + "final_train-scaled.tsv", 10000, testX, testYT, 10000, postprocess=ss)
    joblib.dump(sgd, base + 'sgd-fitted.pkl')
    sgd = joblib.load(base + 'sgd-fitted.pkl')



