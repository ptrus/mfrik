def average_results(outpath, *paths):
    with open(outpath, 'w') as fout:
        targets = []
        first = True
        for inpath in paths:
            cntr = 0
            with open(inpath, 'r') as fin:
                for line in fin:
                    y = float(line.strip())
                    if first:
                        targets.append([y])
                    else:
                        targets[cntr].append(y)
                        cntr+=1
                first = False
        for ys in targets:
            fout.write("{0:.3f}".format(sum(ys)/len(ys)) + '\n')


if __name__ == '__main__':
    # base = "/home/peterus/Downloads/"
    # base = "C:\\Users\\Peter\\Downloads\\ccdm_large.tsv\\"
    base = "D:\\mfrik_data\\"
    base_output = "C:\\mfrik_data\\"
    # train = base+ file + "-preprocessed.tsv"
    train = base + 'cdm_all.tsv-preprocessed-ALL95.tsv'
    predict = base + 'ccdm_test.tsv-preprocessed-ALL95.tsv'

    average_results(base+'avg.txt', base+'predictions_keras.tsv', base+'xa.tsv')

# TODO STACKING