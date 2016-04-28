import utils
import numpy as np

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
    with open(outpath, 'w') as out:
        for lines in utils.read_tsv_online(inpath):
            if first:
                header = lines.pop(0)
                first = False
                out.write('\t'.join(header) + '\n')

            for line in lines:
                if line[0] == "ADLOADINGTIME": continue # TODO FIX SHUFFLE
                if float(line[idx]) >= percentile: continue
                out.write('\t'.join(line) + '\n')
                n_lines += 1

    print "Lines remaining: %d" % (n_lines)

if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"

    shuffled_path = base + "ccdm_large-shuffled.tsv"
    without_outliers = base + "ccdm_large-shuffled-without-outliers.tsv"
    #remove_outliers(shuffled_path, without_outliers, 0)


    # PARSE TIMESTAMPS, discretasize, parse jsons, select fields on: without_outliers