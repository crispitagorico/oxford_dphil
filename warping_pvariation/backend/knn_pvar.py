import os
import numpy as np
import torch
import signatory
from pvar_tools import *
import multiprocessing
import tempfile
import shutil
from scipy.stats import mode
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, accuracy_score


def reparametrization(length, sub_rate, replace):
    a = np.random.choice(range(length), size=int(sub_rate*length), replace=replace)
    a.sort()
    return [0] + a.tolist() + [length-1]

# def processing(j, pvar, dm, X_train, X_test, length, sub_rate, replace, norm, approx):
#     reparam = reparametrization(length, sub_rate, replace=replace)
#     for i in range(len(X_train)):
#         if not approx:
#             dm[i,j] = p_variation_distance(signatory.Path(X_train[i].path[1][:,reparam,:], 1, basepoint=True), 
#                                            signatory.Path(X_test[j].path[1][:,reparam,:], 1, basepoint=True), 
#                                            pvar, norm)
#         else:
#             dm[i,j] = p_variation_distance_approx(signatory.Path(X_train[i].path[1][:,reparam,:], 1, basepoint=True), 
#                                                   signatory.Path(X_test[j].path[1][:,reparam,:], 1, basepoint=True), 
#                                                   pvar, norm)

def processing(j, pvar, dm, X_train, X_test, length, sub_rate, replace, norm, approx):
    for i in range(len(X_train)):
        if not approx:
            dm[i,j] = p_variation_distance(X_train[i], X_test[j], pvar, norm)
        else:
            dm[i,j] = p_variation_distance_approx(X_train[i], X_test[j], pvar, norm)
        
def knn_pvar(x_train, x_test, y_train, y_test, pvar, sub_rate, length, approx, replace=False, norm='l1', n_neighbours=1):

    reparam = reparametrization(length, sub_rate, replace=replace)
    x_train = x_train[:,reparam,:]
    x_test = x_test[:,reparam,:]
        
    X_train = signatory.Path(x_train, int(pvar))
    X_train = [signatory.Path(g.unsqueeze(0), 1, basepoint=True) for g in X_train._signature[0]]
    X_train = [signatory.Path(torch.cat(l.path,1), 1, basepoint=True) for l in X_train]
    
    X_test = signatory.Path(x_test, int(pvar))
    X_test = [signatory.Path(g.unsqueeze(0), 1, basepoint=True) for g in X_test._signature[0]]
    X_test = [signatory.Path(torch.cat(l.path,1), 1, basepoint=True) for l in X_test]
    
    ## Creat a temporary directory and define the array path
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    dm = np.memmap(filename, dtype=float, shape=(len(X_train), len(X_test)), mode='w+')
    
    # compute distances
    Parallel(n_jobs=multiprocessing.cpu_count(), 
             max_nbytes=None, 
             verbose=0)(delayed(processing)(j, 
                                            pvar, 
                                            dm, 
                                            X_train, 
                                            X_test, 
                                            length, 
                                            sub_rate, 
                                            replace, 
                                            norm,
                                            approx,
                                            ) for j in range(len(X_test)))

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
        