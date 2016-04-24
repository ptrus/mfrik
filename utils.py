import numpy as np
import json
import time
from datetime import datetime
import psutil
import gc
from collections import defaultdict

def read_tsv(path, header=True):
    with open(path, 'r') as f:
        data = []
        if header:
            h = f.readline().strip().split('\t')
        for line in f.readlines():
            data.append(line.strip().split('\t'))
    return np.array(data),h if header else np.array(data)

def read_tsv_online(path, header=True, maxmemusage=80):
    with open(path) as f:
        buffer = []
        for line in f:
            if psutil.virtual_memory().percent < maxmemusage:
                buffer.append(line.strip().split('\t'))
            else:
                print (len(buffer))
                print psutil.virtual_memory()
                print "out"
                yield buffer
                del buffer
                buffer = []
                gc.collect()
                print "in"
                print psutil.virtual_memory()
        yield buffer

def file_apply(inpath, outpath, header=True, fn=id):
    with open(outpath, 'w') as fout:
        with open(inpath) as fin:
            for line in fin:
                fout.write(fn(line))

def write_tsv(path, data, header):
    with open(path, 'w') as f:
        f.write('\t'.join(header) + "\n")
        for line in data:
            f.write('\t'.join([str(x) for x in line]) + "\n")

def unique_vals(path, fields):
    first = True
    unique_vals = defaultdict(set)
    print path
    for lines in read_tsv_online(path):
        print "in here"
        if first:
            header = lines.pop(0)
            first = False
            idxs = [header.index(x) for x in fields]

        for line in lines:
            for idx in idxs:
                unique_vals[header[idx]].add(line[idx])
    print "inhere"
    return unique_vals

def distinct_values(data, idxs):
    results = []
    for idx in idxs:
        uq = np.unique(data[:,idx])
        results.append((uq, len(uq)))
    return results

def output_distinct(data, h):
    data = data[1:]
    discreetes = ["ACCOUNTID","CAMPAIGNID","CREATIVEID","PLACEMENTID", "PLATFORMVERSION",
                  "EXTERNALADSERVER", "EXTERNALCREATIVEID", "EXTERNALPLACEMENTID", "EXTERNALSITEID",
                  "EXTERNALSUPPLIERID", "SDK", "NETWORKTYPE", "GEOIP_REGION", "GEOIP_CITY", "GEOIP_AREACODE",
                  "GEOIP_METROCODE", "GEOIP_DMACODE", "UA_MODEL", "UA_OSVERSION", "UA_BROWSERVERSION"]
    discreetesidxs = [h.index(d) for d in discreetes]
    vals = distinct_values(data, discreetesidxs)
    for (i,(res,cnt)) in enumerate(vals):
        print(discreetes[i])
        print("%d distinct values / %d all values" % (cnt, data.shape[0]))
        print("\n\n")

def test_discretasize(data, h):
    datanew, hnew = discretasize(data, h, h.index("PLATFORM"))

    print h
    print hnew
    print data.shape, len(h)
    print datanew.shape, len(hnew)
    print data[:,h.index("PLATFORM")]
    print datanew[:, -4:]

def discretasize(data, header, idx):
    if type(idx) == str:
        idx = header.index(idx)

    col = data[:, idx]
    uniqs = list(np.unique(col))
    new_cols = np.zeros((data.shape[0], len(uniqs)))
    vfunc = np.vectorize(lambda x: uniqs.index(x))
    idxs = vfunc(col)
    for i,idc in enumerate(idxs):
        new_cols[i, idc] = 1
    data = np.delete(data, idx, 1)
    data = np.append(data, new_cols, 1)
    new_headers = [header[idx] + "_" + str(val) for val in uniqs]
    header = list(header)
    del header[idx]
    return data,header + new_headers

def parse_filejson(data, h, idx):
    hnew = h[idx] + "_size"
    hnew2 = h[idx] + "_len"
    sizes = []
    lens = []
    for j in data[:,idx]:
        d = json.loads(j)
        l = len(d)
        s = 0
        for key in d:
            s += key['size']
        sizes.append(s)
        lens.append(l)

    data = np.delete(data, idx, 1)
    sizes = np.array(sizes).reshape(len(sizes), 1)
    data = np.append(data, sizes, 1)
    lens = np.array(lens).reshape(len(lens), 1)
    data = np.append(data, lens, 1)
    del h[idx]
    return data, h + [hnew, hnew2]

# TODO: distinct
def parse_errorjson(data, h, idx):
    hnew = h[idx] + "_len"
    lens = []
    for j in data[:, idx]:
        lens.append(len(json.loads(j)))
    data = np.delete(data, idx, 1)
    lens = np.array(lens).reshape(len(lens),1)
    data = np.append(data, lens, 1)
    del h[idx]
    return data, h + [hnew]

def parse_timestamp(data, h, idx):
    dayofmonth = []
    dayofweek = []
    hour = []
    year = []
    month = []
    minute = []
    for j in data[:, idx]:
        d = datetime.fromtimestamp(float(j))
        dayofmonth.append(d.day)
        dayofweek.append(d.weekday())
        hour.append(d.hour)
        month.append(d.month)
        year.append(d.year)
        minute.append(d.minute)
    dayofmonth = np.array(dayofmonth).reshape(len(dayofmonth), 1)
    dayofweek = np.array(dayofweek).reshape(len(dayofweek), 1)
    hour = np.array(hour).reshape(len(hour), 1)
    month = np.array(month).reshape(len(month), 1)
    year = np.array(year).reshape(len(year), 1)
    minute = np.array(minute).reshape(len(minute), 1)
    data = np.append(data, dayofmonth, 1)
    data = np.append(data, dayofweek, 1)
    data = np.append(data, hour, 1)
    data = np.append(data, month, 1)
    data = np.append(data, year, 1)
    data = np.append(data, minute, 1)
    return data, h + ["timestamp_dayofmonth", "timestamp_dayofweek", "timestamp_hour", "timestamp_month", "timestamp_year", "timestamp_minute"]


def remove_outliers(data, idx):
    #mean = np.mean(data[:,idx].astype(float))
    #var = np.mean(data[:,idx].astype(float))
    devet6 = np.percentile(data[:,idx].astype(float), 96)
    mask = np.array([float(data[i,idx]) < devet6  for i in range(data.shape[0])])
    data = data[mask]
    return data

if __name__ == '__main__':
    '''
    for chunk in read_tsv_online("C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\ccdm_large.tsv"):
        print len(chunk)
        print "in here"
    '''

    ''' unique vals '''
    ALL_CATEGORIES = ["ACCOUNTID","CAMPAIGNID","PLACEMENTID", "CREATIVEID","CREATIVETYPE",

                      "PLATFORM", "PLATFORMVERSION", "INTENDEDDEVICETYPE", "ACTUALDEVICETYPE",

                      "DEVICEORIENTATION", "SDK", "NETWORKTYPE", "CDNNAME",

                      "EXTERNALADSERVER", "EXTERNALCREATIVEID", "EXTERNALPLACEMENTID", "EXTERNALSITEID",
                      "EXTERNALSUPPLIERID",

                      "GEOIP_TIMEZONE", "GEOIP_COUNTRY", "GEOIP_REGION", "GEOIP_CITY", "GEOIP_AREACODE",
                      "GEOIP_METROCODE", "GEOIP_DMACODE",

                      "UA_HARDWARETYPE", "UA_DEVICETYPE",  "UA_MOBILEDEVICE", "UA_PLATFORM", "UA_PLATFORMVERSION",
                      "UA_VENDOR", "UA_MODEL", "UA_OS", "UA_OSVERSION", "UA_BROWSER", "UA_BROWSERVERSION",
                      "UA_BROWSERRENDERINGENGINE"
                     ]

    ALL_CONTINIOUS = ["TOPMOSTREACHABLEWINDOWHEIGHT", "TOPMOSTREACHABLEWINDOWWIDTH", "HOSTWINDOWHEIGHT", "HOSTWINDOWWIDTH]"]

    JSON = ["FILESJSON", "ERRORSJSON"]

    TIMESTAMPS = ["TIMESTAMP"]
    GEO = ["GEOIP_LNG", "GEOIP_LAT"]

    #uv = unique_vals("/home/peterus/Projects/mfrik/ccdm_medium.tsv", ALL_CATEGORIES)
    uv = unique_vals("/home/peterus/Downloads/ccdm_large.tsv", ALL_CATEGORIES)

    for key,val in uv.items():
        print key, len(val)

    END()
    tick = time.time()
    #data,h = read_tsv("D:\\mfrik\\ccdm_01_public_sample.tsv")
    data,h = read_tsv("D:\\mfrik\\ccdm_medium.tsv")
    #data, h = read_tsv("C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\ccdm_large.tsv")
    print 100*(time.time()-tick)
    #output_distinct(data, h)
    #test_discretasize(data,h)
    data = remove_outliers(data, 0)

    data, h = parse_timestamp(data, h, h.index("TIMESTAMP"))

    selected = ["ADLOADINGTIME", "PLATFORM", "INTENDEDDEVICETYPE", "ACTUALDEVICETYPE", "SDK", "DEVICEORIENTATION", "CDNNAME",
                    "CREATIVETYPE", "TIMESTAMP", "HOSTWINDOWHEIGHT", "HOSTWINDOWWIDTH", "TOPMOSTREACHABLEWINDOWHEIGHT",
                "TOPMOSTREACHABLEWINDOWWIDTH", "FILESJSON", "ERRORSJSON", "EXTERNALCREATIVEID", "NETWORKTYPE",
                "timestamp_dayofmonth", "timestamp_dayofweek", "timestamp_hour", "timestamp_month", "timestamp_year",
                "timestamp_minute"]
    selectedidx = [h.index(s) for s in selected]
    data = data[:, selectedidx]
    h = selected
    disc_attrs = ["PLATFORM", "INTENDEDDEVICETYPE", "ACTUALDEVICETYPE", "SDK", "DEVICEORIENTATION", "CDNNAME",
                    "CREATIVETYPE", "EXTERNALCREATIVEID", "NETWORKTYPE", "timestamp_dayofweek"]
    print data.shape
    print h
    for d in disc_attrs:
        print "working %s" % d
        data, h = discretasize(data, h, d)

    print data.shape
    print h

    data, h = parse_filejson(data, h, h.index("FILESJSON"))

    data, h = parse_errorjson(data, h, h.index("ERRORSJSON"))
    print data.shape
    #write_tsv("D:\\mfrik\\out.tsv", data, h)
