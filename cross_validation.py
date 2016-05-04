import os
import numpy as np

def get_train_test_folds_paths(base_path, uuid, prepreocessed='', n_folds = 5):
    folds = [os.path.join(base_path, uuid + '-fold-' + str(i) + prepreocessed) for i in range(1, n_folds+1)]

    for i in range(n_folds):
        test = np.array(folds[i])
        train = np.delete(np.array(folds), i)
        print train, test
        yield train,test

def get_full_train_paths(base_path, uuid, prepreocessed='', n_folds = 5):
    for _, test in get_train_test_folds_paths(base_path, uuid, prepreocessed, n_folds):
        yield test

def cv_predict(online_learner, basepath, uuid, n_folds=5, prepreocessed='', verbose = False):
    outdir = os.path.join(basepath, uuid + '-' + online_learner.name + '/')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print "Starting CV predictions:"
    i = 0
    for trainpath, testpath in get_train_test_folds_paths(basepath, uuid, prepreocessed, n_folds = n_folds):
        i+=1
        if verbose: print "CV: %d / %d" % (i, n_folds)
        new_learner = online_learner.duplicate()
        if verbose: print "Fitting..."
        new_learner.online_fit(trainpath)
        if verbose: print "Transforming..."
        new_learner.online_transform(testpath, os.path.join(outdir, 'predict.tsv'))

    # Final fit
    for _, testpath in get_full_train_paths(basepath, uuid, prepreocessed, n_folds= 5):
        online_learner.online_fit(testpath)

    online_learner.save(os.path.join(outdir, 'learner'))

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