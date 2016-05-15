from collections import Counter
from datetime import datetime
from operator import itemgetter
from utils import NOT_USED_YET
if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"

    file = "cdm_all.tsv"
    #file = "ccdm_test.tsv"

    counters = {}
    #geoips = ['GEOIP_TIMEZONE', 'GEOIP_COUNTRY', 'GEOIP_REGION', 'GEOIP_CITY', 'GEOIP_AREACODE', 'GEOIP_METROCODE', 'GEOIP_DMACODE']
    fields = NOT_USED_YET
    for g in fields:
        counters[g] = Counter()

    timestamps = Counter()
    with open(base+file) as f:
        h = f.readline()
        h = h.strip().split('\t')
        print h
        idx = h.index('TIMESTAMP')
        geoip_idxs = [h.index(g) for g in fields]
        for line in f:
            row = line.strip().split('\t')
            date = datetime.fromtimestamp(int(line.strip().split('\t')[idx]))
            timestamps[str(date.month) + '_' + str(date.day)] += 1
            for i,gidx in enumerate(geoip_idxs):
                temp = row[gidx]
                counters[fields[i]][temp] += 1

    with open('./not_used_stats.txt', 'w') as f:
        for g in fields:
            f.write(g + '=\\\n')
            i = 0
            f.write('{')
            for key, val in sorted(counters[g].items(), key=itemgetter(1), reverse=True):
                i +=1
                f.write('"' + str(key) + '" : ' + str(val) + ', ')
                if i % 10 == 0:
                    f.write('\n')
            f.write('}')
            f.write('\n\n\n')
    #print timestamps
