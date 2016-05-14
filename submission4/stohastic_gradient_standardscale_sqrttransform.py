from cross_validation import cv_predict
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn_utils import OnlineLearner
import numpy as np

if __name__ == '__main__':
    base = "D:\\mfrik_data\\"
    #base="/home/peterus/Projects/mfrik/"
    #base = "C:\\Users\\peteru\\Downloads\\"

    train_file = base + 'cdm_all.tsv-preprocessed.tsv'
    test_file = base + 'ccdm_test.tsv-preprocessed.tsv'

    for file in [train_file, test_file]:
        print "Sqrting file:", file
        with open(file + '_sqrt', 'w') as fout:
            with open(file, 'r') as fin:
                for line in fin:
                    if line.startswith('ADLOADINGTIME'):
                        # header
                        fout.write(line)
                    else:
                        line = line.strip().split('\t')
                        y = line[0]
                        row = np.array(line[1:])
                        row[row == 'null'] = '0'
                        row = row.astype(float)
                        row = np.sqrt(np.abs(row) + (3 / 8))
                        fout.write(y + '\t')
                        row = [str(x) for x in row]
                        row = '\t'.join(row)
                        fout.write(row + '\n')

    train_file = train_file + '_sqrt'
    test_file = test_file + '_sqrt'

    print "Sandard scaling..."
    onlinescaler = OnlineLearner(StandardScaler(), 10000, 'standardscaler')
    print "Fitting..."
    onlinescaler.online_fit([train_file])
    print "Transforming..."
    onlinescaler.online_transform(train_file, str(train_file) + '_standarscaled')
    onlinescaler.online_transform(test_file, str(test_file) + '_standardsacaled')
    #onlinescaler.save(base +'__standardscaler_fitted_all')

    alphas = [0.1,1,2,5,8,10,20]
    for alpha in alphas:
        print 'Training alphas'
        online_learner = OnlineLearner(SGDRegressor(alpha=alpha), 10000, 'standardscaler', finditer=True)
        online_learner.online_fit([str(train_file) + '_standarscaled'])
        online_learner.online_predict(str(test_file) + '_standardsacaled', str(test_file) + '_standardsacaled'+'_alpha_' + str(alpha))
