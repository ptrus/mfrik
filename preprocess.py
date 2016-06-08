import attributes_map
import discrete_values_map as m
import numpy as np
import json
from datetime import datetime
import discrete_values_counts
from random import randint
import linecache


def merge_tsvs(outpath, *paths):
    """
    Merge multiple tsv files into one.
    :param outpath:
    :param paths:
    :return:
    """
    with open(outpath, 'w') as fout:
        first = True
        for inpath in paths:
            with open(inpath, 'r') as fin:
                for line in fin:
                    # If has header.
                    if line[0].isalpha():
                        # Only copy the header from the first file.
                        if first:
                            fout.write(line)
                    else:
                        fout.write(line)
                first = False


def shuffle_file(in_path, out_path, n_lines, header=True):
    """
    Out of memmory shuffle function.
    :param in_path: file that should be shuffeled.
    :param out_path: output where shuffeled file should be written.
    :param n_lines: number of lines of the file.
    :param header: boolean, if True first row gets copied.
    :return:
    """
    with open(out_path, 'w') as out:
        numbers = list(range(n_lines))
        # Copy header
        if header:
            out.write(linecache.getline(in_path, 1))
            n_lines -= 1

        for i in range(n_lines):
            idx = randint(0, n_lines-i-1)
            if header: idx += 1
            n = numbers.pop(idx)
            assert n != 0
            # Write line n, to output
            out.write(linecache.getline(in_path, n+1))


def get_percentile(path, idx=0, percentile=95):
    """
    :param path: path of the training file.
    :param idx: target idx, from which the percentile should be calculated.
    :param percentile: which percentile to get.
    :return: the value of percentile.
    """
    x = []
    with open(path, 'r') as f:
        for line in f:
            # Skip header if it exsists.
            if line.startswith("ADLOADINGTIME"): continue
            x.append(float(line.strip().split('\t')[idx]))

    return np.percentile(x, percentile)


def remove_outliers(inpath, outpath, idx=0):
    """
    Removes outliers.
    :param inpath: path of the file to remove outliers from.
    :param outpath: path of the output file without outleirs.
    :param idx: index of the target variable
    :return: returns the number of remaining samples.
    """

    print "Removing outliers"
    percentile = get_percentile(inpath, idx, 95)
    print "96th percentile %f" % (percentile)

    first = True
    n_lines = 0
    with open(outpath, 'w') as fout:
        with open(inpath, 'r') as fin:
            for line in fin:
                # First line could be a header.
                if first:
                    first = False
                    if line.startswith("ADLOADINGTIME"):
                        fout.write(line)
                        continue
                row = line.strip().split('\t')
                if float(row[idx]) > percentile: continue
                fout.write(line)
                n_lines += 1

    print "Lines remaining: %d" % (n_lines)
    return n_lines


# Helper array for new header, which consists of discretasized attributes.
discrete_headers = ['UA_DEVICETYPE_' + x for x in m.UA_DEVICETYPE] + ['DEVICEORIENTATION_' + x for x in m.DEVICEORIENTATION] +\
        ['UA_BROWSERRENDERINGENGINE_' + x for x in m.UA_BROWSERRENDERINGENGINE] + ['ACTUALDEVICETYPE_' + x for x in m.ACTUALDEVICETYPE] +\
        ['PLATFORM_' + x for x in m.PLATFORM] + ['INTENDEDDEVICETYPE_' + x for x in m.INTENDEDDEVICETYPE] + ['CDNNAME_' + x for x in m.CDNNAME] +\
        ['EXTERNALADSERVER_' + x for x in m.EXTERNALADSERVER] + ['NETWORKTYPE_' + x for x in m.NETWORKTYPE] +\
        ['ACCOUNTID_' + x for x in m.ACCOUNTID] + ['CREATIVETYPE_' + x for x in m.CREATIVETYPE] + ['UA_OS_' + x for x in m.UA_OS] +\
        ['SDK_' + x for x in m.SDK]


def discretasize_line(line, header_old):
    """
    Helper function for binarizing discreete attributes with low amount of categories.
    :param line: one sample
    :param header_old: header of that sample
    :return: new line, with discreete attributes binarized.
    """
    new_line = np.zeros(len(discrete_headers))
    for T in attributes_map.TEST_SET_ALL:
        idx = header_old.index(T)
        val = line[idx]
        new_idx = discrete_headers.index(T + '_' + val)
        new_line[new_idx] = 1

    assert len(discrete_headers) == len(new_line)
    return new_line.tolist()


def binaries(line, header_old):
    """
    Helper function for transforming binary attributes.
    :param line: one sample
    :param header_old: header of that sample
    :return: new line, with binary attributes preprocessed.
    """
    new_line = []
    for b in attributes_map.BINARY:
        idx = header_old.index(b)
        val = line[idx]
        if val != 'null':
            new_line.append(val)
        else:
            new_line.append('0')

    assert len(attributes_map.BINARY) == len(new_line)
    return new_line


def continious(line, header_old):
    """
    Helper function for transforming continious attributes.
    :param line: one sample
    :param header_old: header of that sample
    :return: new line, with continious attributes preprocessed.
    """

    new_line = []
    for b in attributes_map.ALL_CONTINIOUS:
        idx = header_old.index(b)
        val = line[idx]
        if val != 'null':
            new_line.append(val)
        else:
            if b == 'TOPMOSTREACHABLEWINDOWHEIGHT' or b == 'HOSTWINDOWHEIGHT':
                new_line.append('50')
            elif b == 'TOPMOSTREACHABLEWINDOWWIDTH' or b == 'HOSTWINDOWWIDTH':
                new_line.append('320')

    assert len(attributes_map.ALL_CONTINIOUS) == len(new_line)
    return new_line


def json_parse(line, header_old):
    """
    Helper function for transforming JSON attributes.
    :param line: one sample
    :param header_old: header of that sample
    :return: new line, with JSON attributes preprocessed.
    """
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

    assert len(json_headers) == len(new_line)
    return new_line


def timestamps(line, header_old):
    """
    Helper function for preprocessing timestamps
    :param line: one sample
    :param header_old: header of that sample
    :return: new line, with timestamp preprocessed.
    """
    idx = header_old.index('TIMESTAMP')
    val = line[idx]
    d = datetime.fromtimestamp(float(val))
    new_line = [val, d.day, d.weekday(), d.hour, d.minute, d.second, d.microsecond, (((((d.day * 24 + d.hour) * 60) + d.minute) * 60 + d.second) * 1000000) + d.microsecond]
    assert len(attributes_map.TIMESTAMPS) == len(new_line)
    return new_line


def missings(line, header_old):
    """
    Helper function for adding the number of missing fields attribute.
    :param line: one sample
    :param header_old: header of that sample
    :return: new line, with missing attribute.
    """
    cntr = 0
    for v in line:
        if v == 'null':
            cntr += 1
    new_line = [cntr]
    return new_line


def geo(line, header_old):
    """
    Helper function for preprocessing GEO attributes.
    :param line: one sample
    :param header_old: header of that sample
    :return: new line, with GEO attributes preprocessed.
    """
    new_line = []
    latLng = []
    GEO_lat_lng = ["GEOIP_LNG", "GEOIP_LAT"]
    for b in GEO_lat_lng:
        idx = header_old.index(b)
        val = line[idx]
        if val != 'null':
            latLng.append(val)
            new_line.append(val)
        else:
            new_line.append('0')

    if len(latLng) == 2:
        [lat,lng] = latLng
        new_line.append(float(lat) + float(lng))
        new_line.append(float(lat) * float(lng))
        new_line.append(float(lat) - float(lng))
    else:
        new_line.append(0)
        new_line.append(0)
        new_line.append(1000) # A large value that others dont have. Only happens in 20 cases anyway.

    idx = header_old.index('GEOIP_COUNTRY')
    val = line[idx]
    cntries = geoip_countries
    line = np.zeros(len(cntries))
    if val in cntries:
        new_idx = cntries.index(val)
        line[new_idx] = 1
    else:
        line[-1] = 1

    new_line = new_line + line.tolist()

    assert len(attributes_map.GEO) + len(cntries) == len(new_line)
    return new_line


def discretasize_large(line, header_old):
    """
    Helper function for binarizing discreete attributes with large number of categories.
    :param line: one sample
    :param header_old: header of that sample
    :return: new line, with discrete attributes preprocessed.
    """
    fields = attributes_map.CATEGORIES_LARGE
    new_line = []
    for f in fields:
        idx = header_old.index(f)
        val = line[idx]
        field_values = field_values_dict[f]
        zeros = np.zeros(len(field_values))
        if val in field_values:
            new_idx = field_values.index(val)
            zeros[new_idx] = 1
        else:
            zeros[-1] = 1
        new_line = new_line + zeros.tolist()
    return new_line


def discrete_large_headers(fields):
    """returns headers for fields"""
    new_header = []
    for field in fields:
        new_header = new_header + [field + '_' + val for val in field_values_dict[field]]
    return new_header


atleast=10000
def get_field_values(field, atleast=10000):
    '''return categories for attribute "field" that have at least "atleast" samples with that value'''
    return [key for key,c in getattr(discrete_values_counts, field).items() if c >= atleast] + ['OTHER']


"""values for attribute f"""
field_values_dict = {}
for f in attributes_map.CATEGORIES_LARGE:
    field_values_dict[f] = get_field_values(f)


"""geoip_countries"""
geoip_countries = [key for key,c in m.GEOIP_COUNTRY_vals.items() if c >= atleast] + ['OTHER']


def geoip_heades(geoip_countries):
    """geoip_country_headers"""
    return ['GEOIP_COUNTRY_' + x for x in geoip_countries]

'''json headers'''
json_headers = ['FILESJSON_size', 'FILESJSON_len', 'ERRORJSON_len']


def preprocess(inpath, outpath):
    '''
    Function that preprocesses the input file according to rules written in functions.
    :param inpath: file that should be preprocessed
    :param outpath: output file
    '''
    # Set new header.
    header_new = [attributes_map.TARGET] + discrete_headers +  attributes_map.BINARY + attributes_map.ALL_CONTINIOUS + json_headers + attributes_map.TIMESTAMPS + attributes_map.GEO + geoip_heades(geoip_countries) + discrete_large_headers(attributes_map.CATEGORIES_LARGE) + ['missing_count']
    header_old = []
    first = True
    with open(outpath, 'w') as out:
        with open(inpath, 'r') as fin:
            for line in fin:
                line = np.array(line.strip().split('\t'))
                # First select header.
                if first:
                    first = False
                    out.write('\t'.join(header_new) + '\n')
                    header_old = line.tolist()
                    target_idx = header_old.index(attributes_map.TARGET)
                    continue
                '''Preprocess this line'''
                # TARGET
                new_line = [line[target_idx]]
                # Discretes with small number of categoires.
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
                # Discrete large fields
                new_line += discretasize_large(line, header_old)
                # Number of nulls
                new_line += missings(line, header_old)
                ''' Check that line size matches new header size'''
                assert len(new_line) == len(header_new)
                # Write out this line.
                out.write('\t'.join([str(x) for x in new_line]) + '\n')

if __name__ == '__main__':
    # Base derictroy where you have train and test file.
    base = "D:\\mfrik_data\\"

    fileTrain = "ccdm_all.tsv"
    fileTest = "ccdm_test.tsv"

    # Merge medium, large and sample tsv files into "cdm_all"
    merge_tsvs(base + "ccdm_all.tsv", base + "ccdm_medium.tsv",
               base + "ccdm_large.tsv",
               base + "ccdm_sample.tsv")

    # Prepare train paths.
    base_base_train = base + fileTrain
    without_outliers_train = base + fileTrain + "-without-outliers.tsv"
    shuffled_path_train = base + fileTrain + "-without-outliers-shuffled.tsv"
    preprocessed_train = base + fileTrain + "-preprocessed.tsv"

    # Preprocess train file: remove outliers, shuffle, preprocess.
    # Remove outliers from train.
    new_len = remove_outliers(base_base_train, without_outliers_train, 0)
    # Shuffle train file.
    shuffle_file(without_outliers_train, shuffled_path_train, new_len)
    # Preprocess train file.
    preprocess(shuffled_path_train, preprocessed_train)



    # Prepare test paths.
    base_base_test = base + fileTest
    preprocessed_test = base + fileTest + "-preprocessed.tsv"

    # Preprocess test file: only preprocess the attributes.
    preprocess(base_base_test, preprocessed_test)

