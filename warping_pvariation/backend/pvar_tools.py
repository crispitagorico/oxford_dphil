from pvar_backend import *
import numpy as np
import torch

def vector2tensor(sig, depth):
    """Consumes the signature as a np.array of shape (d,) where d is the dimension of the truncated tensor algebra.
       Returns the list of tensors on each level up to truncation depth."""
    return [sig[:,:2]] + [sig[:,2**k:2**k + 2**(k+1)] for k in range(1, depth)]

def vector_norm(sig_x, sig_y, norm='l1'):
    """Computes ||sig_x - sig_y||, where sig_x, sig_y are two signatures np.arrays of shape (d,)"""
    assert norm in ['l1', 'l2'], "norm must be 'l1' or 'l2'."
    if norm == 'l1':
        return np.abs(sig_x - sig_y).sum()
    elif norm == 'l2':
        return np.sqrt(((sig_x - sig_y)**2).sum())

def tensor_norm_k(sig_x_k, sig_y_k, k, p, norm='l1'):
    """Consumes two signature tensors sig_x_k, sig_y_k np.arrays of shape (d_k), where d_k is r**k, with r being 
       the dimension of the underlying vector space and k the level. p is the p-variation of the paths."""
    assert norm in ['l1', 'l2']
    return vector_norm(sig_x_k, sig_y_k, norm)**(float(p)/float(k))

def tensor_norm(sig1, sig2, depth, norm='l1'):
    """norm diff on the truncated tensor algebra"""
    s1 = vector2tensor(sig1, depth)
    s2 = vector2tensor(sig2, depth)
    norms = []
    for k in range(1, depth+1):
        norms.append(vector_norm(s1[d-1], s2[d-1], norm))
    return max(norms)
    
def pairwise_sig_norm(rough_path1, rough_path2, a, b, p, norm='l1'):
    """norm diff on each point of the 2 rough paths"""
    assert norm in ['l1', 'l2']
    depth = math.floor(p)
    s1 = rough_path1.signature(a, b+1)
    s2 = rough_path2.signature(a, b+1)
    s1 = vector2tensor(s1, depth)
    s2 = vector2tensor(s2, depth)
    norms = []
    for k in range(1, depth+1):
        norms.append(tensor_norm_k(s1[k-1], s2[k-1], k, p, norm))
    return norms

def pairwise_sig_norm_approx(rough_path1, rough_path2, a, b, norm='l1'):
    """pseudo-norm on the truncated tensor algebra (doesn't rescale each level)"""
    assert norm in ['l1', 'l2']
    s1 = rough_path1.signature(a, b+1)
    s2 = rough_path2.signature(a, b+1)
    return vector_norm(s1, s2, norm)

def p_variation_distance(rough_path1, rough_path2, p, norm='l1'):
    """returns the exact p-variation distance and optimal partition points between two paths"""
    assert norm in ['l1', 'l2']
    assert rough_path1.signature_size(1) == rough_path2.signature_size(1)
    length = rough_path1.signature_size(1)
    depth = int(p)
    distance = lambda a,b: pairwise_sig_norm(rough_path1, rough_path2, a, b, p, norm)
    pvars = []
    for k in range(depth):
        dist = lambda a,b: distance(a,b)[k] 
        # dynamic programming
        cum_p_var = [0.] * length        
        for j in range(1, length):       
            for r in range(j):
                temp = dist(r, j) + cum_p_var[r]
                if cum_p_var[j] < temp:
                    cum_p_var[j] = temp
        pvars.append(cum_p_var[-1]**(float(k+1)/float(p)))
    return max(pvars)

def p_variation_distance_approx(rough_path1, rough_path2, p, norm='l1'):
    """returns p-variation distance and optimal partition points between two paths
       without taking into account the level rescaling"""
    assert norm in ['l1', 'l2']
    assert rough_path1.signature_size(1) == rough_path2.signature_size(1)
    length = rough_path1.signature_size(1)
    distance = lambda a,b: pairwise_sig_norm_approx(rough_path1, rough_path2, a, b, norm)
    return p_var_backbone_ref(length, p, distance, optim_partition=False)[0]

def p_variation(rough_path, p, norm='l1', optim_partition=False):
    """returns p-variation and optimal partition points of a path up to level
       depth using the given norm and Dynamic Programming algorithm"""
    length = rough_path.signature_length
    depth = int(p)
    dist = lambda a, b: tensor_norm(rough_path.signature(0, b), rough_path.signature(0, a), depth, norm)
    return p_var_backbone_ref(length, p, dist, optim_partition)



