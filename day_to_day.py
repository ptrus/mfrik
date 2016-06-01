from datetime import datetime
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from sklearn_utils import rmse_scorrer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from operator import itemgetter
from utils import rmse
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
from sklearn.externals import joblib

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    base = "d:\\mfrik_data\\"
    #base = "C:\\Users\\peteru\\Downloads\\"

    file = "cdm_all.tsv"
    base_base = base + file
    without_outliers = base + file + "-without-outliers.tsv"
    shuffled_path = base + file + "-without-outliers-shuffled.tsv"
    preprocessed = base+ file + "-preprocessed-with_cntriesALL.tsv"

    used_attrs = ["UA_MODEL_iPhone", "SDK_MobileWeb", "timestamp_second", "timestamp_minute", "GEOIP_LAT",
                  "TOPMOSTREACHABLEWINDOWHEIGHT",
                  "GEOIP_LNG", "timestamp_alteranivestamp", "TIMESTAMP", "timestamp_hour",
                  "TOPMOSTREACHABLEWINDOWWIDTH", "CDNNAME_c",
                  "CDNNAME_b", "HOSTWINDOWHEIGHT", "CDNNAME_a", "EXTERNALSITEID_79a633437974c47dfbe937dbdb28a6b0",
                  "HOSTWINDOWWIDTH",
                  "CREATIVETYPE_Interstitial", "FILESJSON_size", "ACCOUNTID_e509567e0d78b3cd8b2d8e40f570162d",
                  "SDK_6650fadcc0264109f8bd976558377652",
                  "FILESJSON_len", "ERRORJSON_len", "EXTERNALSITEID_e8dfd4d5a2d3f6641f6d61eac1c321cf",
                  "GEOIP_COUNTRY_United States", "UA_BROWSERVERSION_49.0.2623.105",
                  "UA_PLATFORM_Android", "UA_OS_Android", "EXTERNALPLACEMENTID_OTHER",
                  "ACCOUNTID_ce608160cb7578c77d638c1b4a9aacfa", "GEOIP_REGION_OTHER",
                  "GEOIP_AREACODE_OTHER", "UA_BROWSERVERSION_OTHER", "GEOIP_CITY_OTHER", "CREATIVETYPE_Reveal",
                  "CAMPAIGNID_OTHER", "CREATIVEID_OTHER",
                  "PLACEMENTID_OTHER", "CAMPAIGNID_5551d46cc711d64b701aa0d1bc896755", "DEVICEORIENTATION_90",
                  "CREATIVETYPE_Banner", "GEOIP_TIMEZONE_America/New_York",
                  "GEOIP_COUNTRY_OTHER", "UA_BROWSERVERSION_48.0.2564.106",
                  "ACCOUNTID_98631171fb929a05c906945548455230", "EXTERNALADSERVER_149d62d213b2dfec327ac2dc3391d0d5",
                  "UA_VENDOR_Samsung", "GEOIP_TIMEZONE_America/Chicago", "CREATIVETYPE_ExpandableBanner",
                  "GEOIP_REGION_New York", "GEOIP_DMACODE_OTHER", "DEVICEORIENTATION_0",
                  "GEOIP_REGION_null", "GEOIP_METROCODE_OTHER", "INTENDEDDEVICETYPE_Phone", "GEOIP_REGION_Texas",
                  "GEOIP_CITY_null", "UA_VENDOR_LG", "CAMPAIGNID_0d6fc6a0aee47855ef5ad9dc50db0c41",
                  "UA_OSVERSION_OTHER", "EXTERNALPLACEMENTID_10efe8cb24fcd1321ab3a049f4499353",
                  "EXTERNALSITEID_d9c9180c1a8bfd5ad3b6767cb91059fd", "CREATIVETYPE_Interscroller",
                  "GEOIP_REGION_Florida",
                  "GEOIP_REGION_Virginia", "UA_BROWSERVERSION_null", "GEOIP_TIMEZONE_null", "UA_OSVERSION_9.3.1",
                  "PLATFORMVERSION_9.3.1", "UA_PLATFORMVERSION_9.3.1", "GEOIP_REGION_Pennsylvania",
                  "GEOIP_REGION_California",
                  "UA_MODEL_OTHER", "PLATFORMVERSION_9.2.1", "UA_OSVERSION_9.2.1", "UA_PLATFORMVERSION_9.2.1",
                  "EXTERNALSITEID_OTHER", "GEOIP_TIMEZONE_America/Los_Angeles", "UA_BROWSERVERSION_9.0",
                  "UA_OSVERSION_9.3", "UA_PLATFORMVERSION_9.3", "PLATFORMVERSION_9.3", "UA_VENDOR_OTHER",
                  "UA_OSVERSION_NT 6.1", "ACCOUNTID_b3bfec32aeb2724d2ff03da0a19ff6b8", "GEOIP_REGION_Michigan",
                  "ACCOUNTID_47e9af3f9bd919e131062f46f44fba4e", "UA_OSVERSION_6.0.1", "PLATFORMVERSION_6.0.1",
                  "GEOIP_REGION_Wisconsin", "GEOIP_REGION_Georgia", "UA_OSVERSION_5.1.1", "PLATFORMVERSION_5.1.1",
                  "UA_PLATFORMVERSION_5.1.1", "GEOIP_REGION_Ohio",
                  "EXTERNALPLACEMENTID_9a09d7bc00799e13f41019f060683ef2", "UA_BROWSER_Chrome Mobile",
                  "GEOIP_TIMEZONE_America/Denver", "GEOIP_REGION_Illinois", "GEOIP_REGION_Tennessee"]
    used_attr_ids = []


    out = []
    with open(preprocessed, 'r') as f:
        header = f.readline()
        header = header.strip().split('\t')
        t_idx = header.index('TIMESTAMP')
        if used_attr_ids == []:
            for a in used_attrs:
                used_attr_ids.append(header.index(a) - 1)
            print used_attr_ids

        for line in f:
            line = line.strip().split('\t')
            if datetime.fromtimestamp(float(str(line[t_idx]))).day == 4:
                out.append(line)

    out = np.array(out)
    out[out == 'null'] = '0'
    x,y = out[:,1:].astype(float), out[:,0].astype(float)
    x =  x[:,used_attr_ids]
    x_valid = x[-10000:]
    y_valid = y[-10000:]
    x = x[:-10000]
    y = y[:-10000]

    param = {
             'silent': 0,
             'nthread': 7
             }

    dtrain = xgb.DMatrix(x, y)
    dtest = xgb.DMatrix(x_valid, y_valid)
    num_round = 20000
    evallist = [(dtrain, 'train'), (dtest, 'test')]
    plst = param.items()

    bst = xgb.train(plst, dtrain, num_round, evallist, verbose_eval=100, early_stopping_rounds=100)
    xgb.importance(header[1:], model = bst)

    END()

    vt = VarianceThreshold()
    base = sum(y) / len(y)
    base = rmse(y, [base]*len(y))
    print "Baseline score: ", base
    print x.shape
    gbm = xgb.XGBRegressor(n_estimators=1000)
    gbm.fit(x,y, verbose=True) #eval_set=[(x_valid, 'train'), (y_valid,'test')], early_stopping_rounds=10,
    joblib.dump(gbm, './dump1000')
    print gbm.feature_importances_

    importances = gbm.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %d (%f) - %s" % (f + 1, indices[f], importances[indices[f]], header[indices[f] + 1]))

    end()
    '''
    gbm = xgb.XGBRegressor()
    params = dict(n_estimators=[10,20,50,100,200,500,1000])
    clf = GridSearchCV(gbm, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print "XGBoost:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores:
        print score
    '''

    forest = ExtraTreesRegressor(n_estimators=300,
                                 n_jobs=-1)

    forest.fit(x, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %d (%f) - %s" % (f + 1, indices[f], importances[indices[f]], header[indices[f] + 1]))

    ids = [indices[:10], indices[:20], indices[:40], indices[:50], indices[:80], indices[:100], indices[:200]]

    for idx in ids:
        print "Taking indices ",  len(idx)
        for id in idx:
            print header[id + 1]

        xtemp = x[:,idx]
        print xtemp.shape

        print "Shape before:", xtemp.shape
        xtemp = vt.fit_transform(xtemp)
        print "Shape after:", xtemp.shape
        xtemp, ytemp = shuffle(xtemp, y)


        et = ExtraTreesRegressor(n_jobs=-1)
        params = dict(n_estimators = [500])
        clf = GridSearchCV(et, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
        clf.fit(xtemp, ytemp)
        print "ExtraTreesRegressor:"
        scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
        for score in scores:
            print score

    end()


    reg = Ridge()
    params = dict(reg__alpha= [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000])
    ss = StandardScaler()

    p = Pipeline([('ss', ss), ('reg', reg)])
    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print "Ridge normal:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score


    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x2, y)
    print "Ridge sqrt:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score

    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x3, y)
    print "Ridge log:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score
    '''
    svr = SVR(C=1)
    p = Pipeline([('s', s), ('svr', svr)])
    params = dict(svr__C= [0.01, 0.1, 1, 10, 100], svr__kernel=['rbf', 'linear'])
    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x, y)
    print "Svr normal:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score


    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x2, y)
    print "Svr sqrt:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score


    clf = GridSearchCV(p, params, scoring=rmse_scorrer, n_jobs=1, verbose=10, cv=2)
    clf.fit(x3, y)
    print "Svr log:"
    scores = sorted(clf.grid_scores_, key=itemgetter(1), reverse=True)
    for score in scores[:5]:
        print score
    '''
