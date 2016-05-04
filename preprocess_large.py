import utils
import discrete_values_map as m
import numpy as np
import json
from datetime import datetime

# TODO: fix shuffle
def get_percentile(path, idx=0, percentile=96):
    x = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("ADLOADINGTIME"): continue # TODO FIX SHUFFLE
            x.append(float(line.strip().split('\t')[idx]))

    return np.percentile(x, percentile)


def remove_outliers(inpath, outpath, idx=0):
    print "Removing outliers"
    percentile = get_percentile(inpath, idx, 96)
    print "96th percentile %f" % (percentile)

    first = True
    n_lines = 0
    with open(outpath, 'w') as fout:
        with open(inpath, 'r') as fin:
            for line in fin:
                if first:
                    first = False
                    if line.startswith("ADLOADINGTIME"):
                        fout.write(line)
                        continue
                row = line.strip().split('\t')
                if float(row[idx]) >= percentile: continue
                fout.write(line)
                n_lines += 1

    print "Lines remaining: %d" % (n_lines)
    return n_lines

def discreete_headers():
    return ['UA_DEVICETYPE_' + x for x in m.UA_DEVICETYPE] + ['DEVICEORIENTATION_' + x for x in m.DEVICEORIENTATION] +\
        ['UA_BROWSERRENDERINGENGINE_' + x for x in m.UA_BROWSERRENDERINGENGINE] + ['ACTUALDEVICETYPE_' + x for x in m.ACTUALDEVICETYPE] +\
        ['PLATFORM_' + x for x in m.PLATFORM] + ['INTENDEDDEVICETYPE_' + x for x in m.INTENDEDDEVICETYPE] + ['CDNNAME_' + x for x in m.CDNNAME] +\
        ['EXTERNALADSERVER_' + x for x in m.EXTERNALADSERVER] + ['NETWORKTYPE_' + x for x in m.NETWORKTYPE] +\
        ['ACCOUNTID_' + x for x in m.ACCOUNTID] + ['CREATIVETYPE_' + x for x in m.CREATIVETYPE] + ['UA_OS_' + x for x in m.UA_OS] +\
        ['SDK_' + x for x in m.SDK]

def discretasize_line(line, header_old):
    new_line = np.zeros(len(discreete_headers()))
    for T in utils.TEST_SET_ALL:
        idx = header_old.index(T)
        val = line[idx]
        new_idx = discreete_headers().index(T + '_' + val)
        new_line[new_idx] = 1

    assert len(discreete_headers()) == len(new_line)
    return new_line.tolist()

def binaries(line, header_old):
    new_line = []
    for b in utils.BINARY:
        idx = header_old.index(b)
        val = line[idx]
        if val == 'null':
            val = '0'
        new_line.append(val)

    assert len(utils.BINARY) == len(new_line)
    return new_line

def continious(line, header_old):
    new_line = []
    for b in utils.ALL_CONTINIOUS:
        idx = header_old.index(b)
        val = line[idx]
        if val == 'null':
            val = '0'
        new_line.append(val)

    assert len(utils.ALL_CONTINIOUS) == len(new_line)
    return new_line

def json_parse(line, header_old):
    new_line = []
    # FILEJSON
    idx = header_old.index('FILESJSON')
    val = line[idx]
    d = json.loads(val)
    s = 0
    for key in d:
        s += key['size']
    new_line.append(s)
    new_line.append(len(d))
    # ERRORJSON
    idx = header_old.index('ERRORSJSON')
    val = line[idx]
    d = json.loads(val)
    new_line.append(len(d))

    assert len(json_headers()) == len(new_line)
    return new_line

def timestamps(line, header_old):
    idx = header_old.index('TIMESTAMP')
    val = line[idx]
    d = datetime.fromtimestamp(float(val))
    new_line = [val, d.day, d.weekday(), d.hour, d.minute, d.second, (((((d.day * 24 + d.hour) * 60) + d.minute) * 60 + d.second) * 1000000) + d.microsecond]
    assert len(utils.TIMESTAMPS) == len(new_line)
    return new_line

def geo(line, header_old):
    new_line = []
    for b in utils.GEO:
        idx = header_old.index(b)
        val = line[idx]
        if val == 'null':
            val = '0'
        new_line.append(val)

    assert len(utils.GEO) == len(new_line)
    return new_line

def json_headers():
    return ['FILESJSON_size', 'FILESJSON_len', 'ERRORJSON_len']

def split_train_test(in_path, out_path, test_size=50000):
    with open(in_path, 'r') as fin:
        n = 0
        with open(out_path+'_test.tsv', 'w') as fout:
            while n < test_size:
                fout.write(fin.readline())
                n += 1

        with open(out_path+'_train.tsv', 'w') as fout:
            for line in fin:
                fout.write(line)

def preprocess(inpath, outpath):
    # ORDER IS IMOPRTANT
    header_new = [utils.TARGET] + discreete_headers() +  utils.BINARY + utils.ALL_CONTINIOUS + json_headers() + utils.TIMESTAMPS + utils.GEO
    header_old = []
    first = True
    with open(outpath, 'w') as out:
        #for lines in utils.read_tsv_online(inpath, maxmemusage=80):
        with open(inpath, 'r') as fin:
            for line in fin:
                line = np.array(line.strip().split('\t'))
                if first:
                    first = False
                    out.write('\t'.join(header_new) + '\n')
                    header_old = line.tolist()
                    print header_old
                    target_idx = header_old.index(utils.TARGET)
                    continue
                # TARGET
                new_line = [line[target_idx]]
                # Discreets
                new_line += discretasize_line(line, header_old)
                # Binary
                new_line += binaries(line, header_old)
                # Continious
                new_line += continious(line, header_old)
                # Json
                new_line += json_parse(line, header_old)
                # Timestamps
                new_line += timestamps(line, header_old)
                # Geo
                new_line += geo(line, header_old)
                assert len(new_line) == len(header_new)
                out.write('\t'.join([str(x) for x in new_line]) + '\n')

from os import path
import uuid
from shuffle import shuffle_file

def create_folds(in_path, out_path, n_folds, n_lines):
    f_length = int(n_lines / n_folds)
    f_lengths = [f_length] * n_folds
    if (sum(f_lengths) != n_lines):
        f_lengths[-1] += (n_lines - sum(f_lengths))

    print f_lengths
    uid = str(uuid.uuid4())
    hcount = 0
    with open(in_path) as fin:
        cnt = 0
        for i in f_lengths:
            cnt += 1
            with open(path.join(out_path, uid+'-fold-' + str(cnt)), 'w') as fout:
                j = 0
                while j < i:
                    line = fin.readline()
                    if line and line[0].isalpha(): # Header
                        hcount += 1
                        continue
                    j += 1
                    fout.write(line)

    assert hcount <= 1

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "C:\\Users\\peteru\\Downloads\\"


    base_base = base + "cdm_all.tsv"
    shuffled_path = base + "ccdm_large-shuffled.tsv"
    without_outliers = base + "ccdm_large-shuffled-without-outliers.tsv"
    preprocessed = base+"ccdm_large_preprocessed.tsv"

    new_len = remove_outliers(base_base, without_outliers, 0)
    shuffle_file(without_outliers, shuffled_path, new_len)
    # PARSE TIMESTAMPS, discretasize, parse jsons, select fields

    preprocess(shuffled_path, preprocessed)
    # Create folds
    create_folds(preprocessed, base, 5, new_len)