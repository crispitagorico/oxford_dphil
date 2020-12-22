import math
import collections
import random
import time
import copy
import numpy as np

def p_var_backbone_ref(path_size, p, path_dist, optim_partition=False):
    # p-variation via Dynamic Programming: it maximizes the sum of norm
    # of increments over partitions of the parametrization I
    #
    # input: 
    #        path_size (int) size of I
    #        p (float) p-variation param
    #        path_dist (obj) this is an input distance that takes in 2 indices i,j \in I and returns a scalar
    #        optim_partition (bool) indicates whether the optimal parition should be computed or not
    #
    # returns: 
    #        distance (float), optimal partition (list of tuples, optional)
    

    # check path is not degenerate
    if path_size == 0:
        return -math.inf
    elif path_size == 1:
        return 0

    # setup values
    cum_p_var = [0.] * path_size
    # set up partition
    point_links = [0] * path_size

    # dynamic programming
    for j in range(1, path_size):       
        for k in range(j):
            temp = path_dist(k, j)**p + cum_p_var[k]
            if cum_p_var[j] < temp:
                cum_p_var[j] = temp
                point_links[j] = k
    
    # return also the optimal partition
    if optim_partition:
        points = []
        point_i = path_size-1
        while True:
            points.append(point_i)
            if point_i == 0:
                break
            point_i = point_links[point_i]
        points.reverse()
    else:
        points = []

    return cum_p_var[-1]**(1./p), points
