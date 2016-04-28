from utils import read_tsv_batch
from sklearn.linear_model import SGDRegressor
from sklearn_utils import rmse_scorrer

def SGD_fit(train_path, n_iters, conv_test_x=[], conv_test_y=[], batchsize=1000, loss='squared_loss', penalty='l2', alpha=0.00001, l1_ratio=0.15):
    sgd = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio)
    old_score = None
    iters = 0
    while iters < n_iters:
        iters += 1
        for batch in read_tsv_batch(train_path, first_line=False, batchsize=batchsize):
            x, y = batch[:, 1:], batch[:, 0]
            x[x == 'null'] = '0'
            x = x.astype(float)
            y = y.astype(float)
            sgd.partial_fit(x,y)

        if conv_test_y != [] and n % 10 == 0:
            print "In iteration: %d" % (iters)
            if old_score is None:
                old_score = rmse_scorrer(sgd, conv_test_x, conv_test_y)
                continue
            score = rmse_scorrer(sgd, conv_test_x, conv_test_y)
            print "Old score: %2.f, new score: %2.f" % (old_score, score)
            if score > old_score:
                break
    print "FInished after %d iters." % (iters)

    return sgd

if __name__ == '__main__':
    base = "D:\\mfrik\\"
    #base="/home/peterus/Projects/mfrik/"

    SGD_fit(base + "outALL.tsv", 100)

