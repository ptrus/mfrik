from os import path
import numpy as np

def get_train_test_folds(base_path, uuid, n_folds = 5):
    folds = [path.join(base_path, uuid + '-fold-' + str(i)) for i in range(1, n_folds+1)]

    for i in range(n_folds):
        test = np.array(folds[i])
        train = np.array(folds).delete(i)
        yield train,test


if __name__ == '__main__':
    #base =
    #uuid =
    '''
    for train,test in get_train_test_folds(base, uuid):
        new_est
        for t in train:
            train(t)
        predict(test)
        write to file('append')
    '''