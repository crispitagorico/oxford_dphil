import os
import numpy as np
import torch
import signatory
from tslearn.metrics import dtw
from knn_pvar import reparametrization
import multiprocessing
import tempfile
import shutil
from scipy.stats import mode
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score


def processing(j, dm, x_train, x_test, length, sub_rate, replace):
    reparam = reparametrization(length, sub_rate, replace=replace)
    for i in range(len(x_train)):
        dm[i,j] = dtw(x_train[i][reparam,:], x_test[j][reparam,:])
        

def knn_dtw(x_train, x_test, y_train, y_test, sub_rate, length, replace=False, n_neighbours=1):
    
    ## Creat a temporary directory and define the array path
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    dm = np.memmap(filename, dtype=float, shape=(len(x_train), len(x_test)), mode='w+')
    
    # compute distances
    Parallel(n_jobs=multiprocessing.cpu_count(), 
             max_nbytes=None, 
             verbose=0)(delayed(processing)(j, 
                                            dm, 
                                            x_train, 
                                            x_test, 
                                            length, 
                                            sub_rate, 
                                            replace,
                                            ) for j in range(len(x_test)))

    #Delete the temporary directory and contents
    try:
        shutil.rmtree(temp_folder)
    except OSError:
        pass
    
    knn_idx = dm.T.argsort()[:, :n_neighbours]
    knn_labels = y_train[knn_idx]

    mode_data = mode(knn_labels, axis=1)
    mode_label = mode_data[0]
    mode_proba = mode_data[1]/n_neighbours

    label = mode_label.ravel()
    proba = mode_proba.ravel()
    
    conf_mat = confusion_matrix(label, y_test)
    conf_mat = conf_mat/conf_mat.sum(0)
    
    acc_score = accuracy_score(label, y_test)

    return label, proba, acc_score, conf_mat