import numpy as np
import signatory
import iisignature
import torch
import math
import string
import ast

def Ito_map(X):
    N = X.shape[1]
    Y = torch.zeros((1, N, 3))#.cuda()
    if X.shape[2] == 2:
        Y[0,:,0] = X[0,:,0]
        Y[0,:,1] = X[0,:,1] 
        Y[0,:,2] = torch.tensor([0.] + [signatory.logsignature(X[:,:i+1,:], 2, mode='brackets')[0,-1] for i in range(1, N)])#.cuda()
    else:
        Y[0,:,0] = X[0,:,0]
        Y[0,:,1] = X[0,:,1]
        Y[0,:,2] = 0.5*(X[0,:,3] - X[0,:,4])
    return Y

p = iisignature.prepare(2,2)

def Ito_map_iisig(x):
    if isinstance(x, list):
        xx = x[0]
        x_eps = x[1]
        X = np.stack([xx.reshape(-1), x_eps[:len(xx)]]).T
    else:
        X = x
    N = len(X)
    Y = np.zeros((N, 3))
    if X.shape[1] == 2:
        Y[:,0] = X[:,0]
        Y[:,1] = X[:,1] 
        Y[:,2] = np.array([iisignature.logsig(X[:i,:], p)[-1] for i in range(1, N+1)])
    else:
        Y[:,0] = X[:,0]
        Y[:,1] = X[:,1]
        Y[:,2] = 0.5*(X[:,3] - X[:,4])
    return Y


def convert(string):
    li = list(string.split(" "))[2:]
    list_tup = []
    for k in li:
        obj = ast.literal_eval(k)
        if not isinstance(obj, tuple):
            list_tup.append((obj,))
        else:
            list_tup.append(obj)
    return list_tup

def split_sig_into_levels(sig, depth=2):
    return [sig[:2]] + [sig[2**k : 2**k+2**(k+1)] for k in range(1, depth)]

def split_keys_into_levels(sig_keys, depth=2):
    keys_list = convert(sig_keys) 
    return [keys_list[:2]] + [keys_list[2**k:2**k + 2**(k+1)] for k in range(1, depth)]

def get_sig_levels_as_tensors(sig):
    levels_list = split_sig_into_levels(sig)
    tensor_list = []
    level = 1
    for sig_k in levels_list:
        tensor_list.append(sig_k.reshape(tuple(level*[2])))
        level += 1
    return tensor_list

def product (first, *rest):
    def loop (acc, first, *rest):
        if not rest:
            for x in first:
                yield (*acc, x)
        else:
            for x in first:
                yield from loop ((*acc, x), *rest)
    return loop ((), first, *rest)

def level_loops():
    return {2 : product(range(2), range(2)), 
            3 : product(range(2), range(2), range(2)),
            4 : product(range(2), range(2), range(2), range(2)),
            5 : product(range(2), range(2), range(2), range(2), range(2)),
            6 : product(range(2), range(2), range(2), range(2), range(2), range(2)),
            7 : product(range(2), range(2), range(2), range(2), range(2), range(2), range(2)), 
            8 : product(range(2), range(2), range(2), range(2), range(2), range(2), range(2), range(2))}

def vector_field_action(sig_tensor, level, B):
    if level==1:
        vector_field = B
    else:
        shape = tuple(level*[2] + [3, 3])
        vector_field = np.zeros(shape)
        for p in level_loops()[level]:
            BB = np.eye(3,3)
            for pp in p:
                BB = BB @ B[pp]
            vector_field[p] = BB
    ind1 = string.ascii_lowercase[:level+2]
    ind2 = string.ascii_lowercase[:level]
    ind3 = ind1[level:]
    einsum_string = f'{ind1}, {ind2} -> {ind3}' 
    return np.einsum(einsum_string, vector_field, sig_tensor) 

def flow_expansion(B, sig, R):
    output = np.eye(3, 3)
    sig_tensors = get_sig_levels_as_tensors(sig)
    level = 1
    for tensor in sig_tensors:
        effect = vector_field_action(tensor, level, B)
        factorial = 1./(float(R)**level)
        output += factorial*effect
        level += 1
    return output

def Picard_RDE_Solver(RoughPath, n, B, R, Y0):
    assert n < len(RoughPath)
    increments = RoughPath[n,:2] - RoughPath[0,:2]
    Y = np.append(Y0[:-1] + increments, Y0[-1])
    sig = RoughPath[n]
#     sig_keys = esig.sigkeys(2, 2)
    expansion = flow_expansion(B, sig, R)
    return expansion.dot(Y)

def flow(RoughPath, B, R, Y0):
    flow = []
    for t in range(len(RoughPath)):
        flow.append(Picard_RDE_Solver(RoughPath, n=t, B=B, R=R, Y0=Y0))
    return np.array(flow)

# A1 = np.array([[0,0,1], [0,0,0], [-1,0,0]], dtype=np.float)
# A2 = np.array([[0,0,0], [0,0,1], [0,-1,0]], dtype=np.float)
# B = np.array((A1, A2))
# R = 1.95
# Y0 = np.array([0.,0.,R])