import numpy as np

def read_tsv(path, header=True):
    with open(path, 'r') as f:
        data = []
        if header:
            h = f.readline().strip().split('\t')
        for line in f.readlines():
            data.append(line.strip().split('\t'))
    return np.array(data),h if header else np.array(data)

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

if __name__ == '__main__':
    #data,h = read_tsv("/home/peterus/Projects/mfrik/ccdm_01_public_sample.tsv")
    data,h = read_tsv("/home/peterus/Projects/mfrik/ccdm_medium.tsv")
    #output_distinct(data, h)
    #test_discretasize(data,h)
    selected = ["PLATFORM", "INTENDEDDEVICETYPE", "ACTUALDEVICETYPE", "SDK", "DEVICEORIENTATION", "CDNNAME",
                    "CREATIVETYPE", "TIMESTAMP", "HOSTWINDOWHEIGHT", "HOSTWINDOWWIDTH", "TOPMOSTREACHABLEWINDOWHEIGHT",
                "TOPMOSTREACHABLEWINDOWWIDTH", "FILESJSON", "ERRORSJSON", "EXTERNALCREATIVEID", "NETWORKTYPE"]
    selectedidx = [h.index(s) for s in selected]
    data = data[:, selectedidx]
    h = selected
    disc_attrs = ["PLATFORM", "INTENDEDDEVICETYPE", "ACTUALDEVICETYPE", "SDK", "DEVICEORIENTATION", "CDNNAME",
                    "CREATIVETYPE", "EXTERNALCREATIVEID", "NETWORKTYPE"]
    print data.shape
    print h
    for d in disc_attrs:
        print "working %s" % d
        data, h = discretasize(data, h, d)

    print data.shape
    print h
