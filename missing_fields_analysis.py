# TODO
# Go through features, observe missing fields
# Go through samples, observe missing fields



import numpy as np
from collections import Counter,defaultdict
if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"

    file = "cdm_all.tsv"
    #file = "ccdm_test.tsv"

    errors = Counter()
    vals = defaultdict(list)
    with(open(base+file, 'r')) as fin:
        header = np.array(fin.readline().strip().split('\t'))
        for line in fin:
            row = np.array(line.strip().split('\t'))
            n_nuls = len(row[row == 'null'])
            if n_nuls > 0:
                if n_nuls > 30:
                    print n_nuls, len(row)

                for h in header[row == 'null']:
                    errors[h] += 1

    print errors

'''

Counter({'EXTERNALCREATIVEID': 2499999, 'NETWORKTYPE': 2499965, 'EXTERNALSUPPLIERID': 2296120,  <- removed

        'UA_BROWSERVERSION': 845773, 'EXTERNALSITEID': 827673, 'EXTERNALPLACEMENTID': 763710,
        'GEOIP_METROCODE': 646553, 'GEOIP_DMACODE': 646553, 'GEOIP_AREACODE': 646394,
        'UA_PLATFORMVERSION': 278130, 'PLATFORMVERSION': 277927, 'UA_MOBILEDEVICE': 276149,
        'GEOIP_CITY': 238692, 'GEOIP_REGION': 169473, 'EXTERNALADSERVER': 145326,
        'GEOIP_TIMEZONE': 96082, 'ACTUALDEVICETYPE': 18242, 'UA_DEVICETYPE': 18175,
        'UA_VENDOR': 9538, 'UA_OSVERSION': 8339, 'UA_MODEL': 6103, 'UA_OS': 5554,
        'UA_BROWSERRENDERINGENGINE': 3319, 'UA_BROWSER': 3075, 'UA_HARDWARETYPE': 2805,
        'UA_PLATFORM': 1732, 'PLATFORM': 643, 'GEOIP_LNG': 351, 'GEOIP_LAT': 270, 'DEVICEORIENTATION': 238,
        'HOSTWINDOWHEIGHT': 127, 'TOPMOSTREACHABLEWINDOWHEIGHT': 127,
        'TOPMOSTREACHABLEWINDOWWIDTH': 127, 'HOSTWINDOWWIDTH': 127, 'GEOIP_COUNTRY': 5})

'''