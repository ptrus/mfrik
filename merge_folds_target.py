# TODO: Add this to the preprocessing when folds are created.
from cross_validation import get_train_test_folds_paths
if __name__ == '__main__':
    base = "D:\\mfrik_data\\stacking\\"
    uuid = "d915135f-c72a-499f-8697-e6c0834addf9"
    folds_target = base + "d915135f-c72a-499f-8697-e6c0834addf9-target"

    with open(folds_target, 'w') as fout:
        for trainpaths, testpath in get_train_test_folds_paths(base, uuid, n_folds=5):
            with open(testpath, 'r') as fin:
                header = fin.readline().strip().split('\t')
                for line in fin:
                    line = line.strip().split('\t')
                    y = float(line[0])
                    fout.write("{0:.3f}".format(y) + '\n')