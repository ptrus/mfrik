from datetime import datetime
from collections import Counter
import numpy as np
from sklearn.externals import joblib
from utils import read_tsv_batch
import pickle
import xgboost as xgb

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"
    base_output = "C:\\mfrik_data\\"
    #train = base+ file + "-preprocessed.tsv"
    train = base + 'cdm_all.tsv-preprocessed-with_cntriesALL.tsv'
    predict = base + 'ccdm_test.tsv-preprocessed-with_cntriesALL.tsv'

    ssum = 0
    count = 0


    days = Counter()
    with open(train, 'r') as fin:
        header = fin.readline()
        header = header.strip().split('\t')
        t_idx = header.index('TIMESTAMP')
        for line in fin:
            line = line.strip().split('\t')
            date = datetime.fromtimestamp(float(str(line[t_idx])))
            days[(str(date.month) + '_' + str(date.day))] += 1
            count += 1
            ssum += float(line[0])

    xm1 = [(key,val) for key,val in days.items() if val > 10000]
    xm2 = [(key,val) for key,val in days.items() if val <= 10000]
    print xm2, sum((val) for _, val in xm2)

    avg = ssum/count
    print "Average:", avg

    for key,_ in xm1:
        #if key not in ['4_2', '4_3', '4_']
        print "At:", key
        gbm = xgb.XGBRegressor(n_estimators=1000)
        [month, day] = key.split('_')
        data = []
        print "Reading input"
        with open(train) as fin:
            header = fin.readline()
            header = header.strip().split('\t')
            print header
            t_idx = header.index('TIMESTAMP')
            for line in fin:
                line = line.strip().split('\t')
                date = datetime.fromtimestamp(float(str(line[t_idx])))
                if str(date.month) == month and str(date.day) == day:
                    data.append(line)
        data = np.array(data)
        data[data == 'null'] = '0'
        data = data.astype(float)
        x,y = data[:,1:], data[:,0]
        #x = x[:, used_attr_ids]
        print x.shape
        print "Fitting"
        gbm.fit(x, y)
        joblib.dump(gbm, base_output + "day_to_day_models/" + key + "-et")

    models = ['4_6', '4_5', '4_4', '4_7', '4_1', '4_3', '4_2', '4_9', '4_8', '4_14', '4_15', '4_10', '4_11', '4_12', '4_13']
    files = {}

    for model in models:
        x = open(base+model+'_test.tsv', 'a')
        files[model] = x
    rest = open(base+'REST_test.tsv', 'a')

    print "Writing to new files"
    with open(predict) as fin:
        header = fin.readline()
        header = header.strip().split('\t')
        t_idx = header.index('TIMESTAMP')
        for i,text in enumerate(fin):
            line = text.strip().split('\t')
            date = datetime.fromtimestamp(float(str(line[t_idx])))
            key = str(date.month) + '_' + str(date.day)
            if key not in models:
                rest.write(str(i) + '\t')
                rest.write(text)
            else:
                files[key].write(str(i) + '\t')
                files[key].write(text)


    avg = 3.92983617223

    print 'Predicting'
    predictions = {}
    for model in models:
        print "model:", model
        gbm = joblib.load(base_output + "day_to_day_models/" + model + "-et")
        print "Loaded models"

        for batch in read_tsv_batch(base+model+'_test.tsv', first_line=True, batchsize=10000):
            x = batch[:,2:]
            x[x == 'null'] = '0'
            x = x.astype(float)
            #x = x[:, used_attr_ids]
            ids = batch[:,0].astype(int)

            pred = gbm.predict(x)
            for i,y in zip(ids, pred):
                predictions[i] = "{0:.3f}".format(y)

        del pred
        del x
        del gbm

    file = open(base+'predictions_dict', 'wb')
    print "Predicing rest"
    with open(base + 'REST_test.tsv', 'r') as fin:
        ids = []
        for line in fin:
            line = line.strip().split('\t')
            ids.append(int(line[0]))

        for i in ids:
            predictions[i] = "{0:.3f}".format(avg)
    pickle.dump(predictions, file)

    f = open(base + 'predictions_dict')
    predictions = pickle.load(f)

    print len(predictions)
    print "writing to file"
    with open(base + 'day_to_day_predictions.tsv', 'w') as fout:
        for i in range(2341137):
            if i in predictions:
                fout.write(predictions[i] + '\n')
            else:
                fout.write(str(3.929) + '\n')
'''
Counter({'4': 2340166, '3': 389, '1': 245, '5': 124, '2': 66, '6': 38, '7': 23, '8': 20, '12': 19, '9': 19, '10': 18, '11': 10})
'''