from datetime import datetime
from collections import Counter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.externals import joblib
from utils import read_tsv_batch
import pickle
if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"

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

    #models = {}

    used_attrs = ["UA_MODEL_iPhone", "SDK_MobileWeb", "timestamp_second", "timestamp_minute", "GEOIP_LAT", "TOPMOSTREACHABLEWINDOWHEIGHT",
                  "GEOIP_LNG", "timestamp_alteranivestamp", "TIMESTAMP", "timestamp_hour", "TOPMOSTREACHABLEWINDOWWIDTH", "CDNNAME_c",
                  "CDNNAME_b", "HOSTWINDOWHEIGHT", "CDNNAME_a", "EXTERNALSITEID_79a633437974c47dfbe937dbdb28a6b0", "HOSTWINDOWWIDTH",
                  "CREATIVETYPE_Interstitial", "FILESJSON_size", "ACCOUNTID_e509567e0d78b3cd8b2d8e40f570162d", "SDK_6650fadcc0264109f8bd976558377652",
                  "FILESJSON_len", "ERRORJSON_len", "EXTERNALSITEID_e8dfd4d5a2d3f6641f6d61eac1c321cf", "GEOIP_COUNTRY_United States", "UA_BROWSERVERSION_49.0.2623.105",
                  "UA_PLATFORM_Android", "UA_OS_Android", "EXTERNALPLACEMENTID_OTHER", "ACCOUNTID_ce608160cb7578c77d638c1b4a9aacfa", "GEOIP_REGION_OTHER",
                  "GEOIP_AREACODE_OTHER", "UA_BROWSERVERSION_OTHER", "GEOIP_CITY_OTHER", "CREATIVETYPE_Reveal", "CAMPAIGNID_OTHER", "CREATIVEID_OTHER",
                  "PLACEMENTID_OTHER", "CAMPAIGNID_5551d46cc711d64b701aa0d1bc896755", "DEVICEORIENTATION_90", "CREATIVETYPE_Banner", "GEOIP_TIMEZONE_America/New_York",
                  "GEOIP_COUNTRY_OTHER", "UA_BROWSERVERSION_48.0.2564.106", "ACCOUNTID_98631171fb929a05c906945548455230", "EXTERNALADSERVER_149d62d213b2dfec327ac2dc3391d0d5",
                  "UA_VENDOR_Samsung", "GEOIP_TIMEZONE_America/Chicago", "CREATIVETYPE_ExpandableBanner", "GEOIP_REGION_New York", "GEOIP_DMACODE_OTHER", "DEVICEORIENTATION_0",
                  "GEOIP_REGION_null", "GEOIP_METROCODE_OTHER", "INTENDEDDEVICETYPE_Phone", "GEOIP_REGION_Texas", "GEOIP_CITY_null", "UA_VENDOR_LG", "CAMPAIGNID_0d6fc6a0aee47855ef5ad9dc50db0c41",
                  "UA_OSVERSION_OTHER", "EXTERNALPLACEMENTID_10efe8cb24fcd1321ab3a049f4499353", "EXTERNALSITEID_d9c9180c1a8bfd5ad3b6767cb91059fd", "CREATIVETYPE_Interscroller", "GEOIP_REGION_Florida",
                  "GEOIP_REGION_Virginia", "UA_BROWSERVERSION_null", "GEOIP_TIMEZONE_null", "UA_OSVERSION_9.3.1", "PLATFORMVERSION_9.3.1", "UA_PLATFORMVERSION_9.3.1", "GEOIP_REGION_Pennsylvania", "GEOIP_REGION_California",
                  "UA_MODEL_OTHER", "PLATFORMVERSION_9.2.1", "UA_OSVERSION_9.2.1", "UA_PLATFORMVERSION_9.2.1", "EXTERNALSITEID_OTHER", "GEOIP_TIMEZONE_America/Los_Angeles", "UA_BROWSERVERSION_9.0",
                  "UA_OSVERSION_9.3", "UA_PLATFORMVERSION_9.3", "PLATFORMVERSION_9.3", "UA_VENDOR_OTHER", "UA_OSVERSION_NT 6.1", "ACCOUNTID_b3bfec32aeb2724d2ff03da0a19ff6b8", "GEOIP_REGION_Michigan",
                  "ACCOUNTID_47e9af3f9bd919e131062f46f44fba4e", "UA_OSVERSION_6.0.1", "PLATFORMVERSION_6.0.1", "GEOIP_REGION_Wisconsin", "GEOIP_REGION_Georgia", "UA_OSVERSION_5.1.1", "PLATFORMVERSION_5.1.1",
                  "UA_PLATFORMVERSION_5.1.1", "GEOIP_REGION_Ohio", "EXTERNALPLACEMENTID_9a09d7bc00799e13f41019f060683ef2", "UA_BROWSER_Chrome Mobile", "GEOIP_TIMEZONE_America/Denver", "GEOIP_REGION_Illinois", "GEOIP_REGION_Tennessee"]
    used_attr_ids = []

    for key,_ in xm1:
        print "At:", key
        et = ExtraTreesRegressor(n_estimators=500, n_jobs=-1)
        [month, day] = key.split('_')
        data = []
        print "Reading input"
        with open(train) as fin:
            header = fin.readline()
            header = header.strip().split('\t')
            print header
            if used_attr_ids == []:
                for a in used_attrs:
                    used_attr_ids.append(header.index(a) -1)
                #used_attrs_ids = [(header.index(used_attr) - 1) for used_attr in used_attrs]
                print used_attr_ids
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
        x = x[:, used_attr_ids]
        print x.shape
        print "Fitting"
        et.fit(x, y)
        joblib.dump(et, base + "day_to_day_models/" + key + "-et")

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
        et = joblib.load(base + "day_to_day_models/" + model + "-et")
        print "Loaded models"

        for batch in read_tsv_batch(base+model+'_test.tsv', first_line=True, batchsize=10000):
            x = batch[:,2:]
            x[x == 'null'] = '0'
            x = x.astype(float)
            x = x[:, used_attr_ids]
            ids = batch[:,0].astype(int)

            pred = et.predict(x)
            for i,y in zip(ids, pred):
                predictions[i] = "{0:.3f}".format(y)

        del pred
        del x
        del et

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