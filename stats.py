
if __name__ == '__main__':
    base = "D:\\mfrik_data\\"

    file = "cdm_all.tsv"

    inpath = base + file
    cols = 0
    rows = 0
    with open(inpath, 'r') as fin:
        cols = len(fin.readline().strip().split('\t'))
        rows += 1
        for line in fin:
            rows += 1

    print rows,cols