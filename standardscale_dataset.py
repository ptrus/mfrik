from sklearn_utils import StandardScaler_online_fit, StandarScaler_online_transform, OnlineLearner
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"
    base_output = "C:\\mfrik_data\\"
    # train = base+ file + "-preprocessed.tsv"
    train = base + 'cdm_all.tsv-preprocessed-ALL95.tsv'
    predict = base + 'ccdm_test.tsv-preprocessed-ALL95.tsv'

    print "Sandard scaling..."
    onlinescaler = OnlineLearner(StandardScaler(), 10000, 'standardscaler')
    print "Fitting..."
    onlinescaler.online_fit([train])
    print "Transforming..."
    onlinescaler.online_transform(train, train + '_standarscaled')
    onlinescaler.online_transform(predict, predict + '_standardsacaled')
