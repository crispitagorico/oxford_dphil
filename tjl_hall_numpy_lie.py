"""
Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurko and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)
"""
import doctest
from tjl_dense_numpy_tensor import rescale
from tjl_dense_numpy_tensor import tensor_add
import numpy as np
from collections import defaultdict
import tjl_dense_numpy_tensor



try:
    from functools import lru_cache
except:
    from functools32 import lru_cache


#try:
#	# Custom memoization
#	from decorators import lru_cache


#    #from functools import lru_cache
#except:
#	# If Python 2.7
#    from functools32 import lru_cache

from tjl_dense_numpy_tensor import blob_size

scalar_type = float


@lru_cache(maxsize=0)
def hall_basis(width, desired_degree=0):
    """
hall_basis(1, 0)
(array([[0, 0]]), array([0]), array([1]), defaultdict(<class 'int'>, {}), 1)
hall_basis(1, 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".\tjl_hall_numpy_lie.py", line 41, in hall_basis
    if ((degrees[i] + degrees[j] == d) & (hall_set[j][0] <= i)):
IndexError: list index out of range
hall_basis(1, 1)
(array([[0, 0],
       [0, 1]]), array([0, 1]), array([1, 1]), defaultdict(<class 'int'>, {(0, 1): 1}), 1)
hall_basis(2,3)
(array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 2],
       [1, 3],
       [2, 3]]), array([0, 1, 1, 2, 3, 3]), array([1, 1, 3, 4]), defaultdict(<class 'int'>, {(0, 1): 1, (0, 2): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}), 2)

    """
    degrees = []
    hall_set = []
    degree_boundaries = []
    reverse_map = defaultdict(int)

    # the first entry in hall_set is not part of the basis but instead is
    # the nominal parent of all self parented elements (letters)
    # its values can be used for wider information about the lie element
    curr_degree = 0
    degrees.append(0)
    p = (0, 0)
    hall_set.append(p)
    degree_boundaries.append(1)
    if desired_degree > 0:
        # level 1 the first basis terms
        degree_boundaries.append(1)
        for i in range(1, width + 1):
            hall_set.append((0, i))
            degrees.append(1)
            reverse_map[(0, i)] = i
        curr_degree += 1
        for d in range(curr_degree + 1, desired_degree + 1):
            bound = len(hall_set)
            degree_boundaries.append(bound)
            for i in range(1, bound + 1):
                for j in range(i + 1, bound + 1):
                    if (degrees[i] + degrees[j] == d) & (hall_set[j][0] <= i):
                        hall_set.append((i, j))
                        degrees.append(d)
                        reverse_map[(i, j)] = len(hall_set) - 1
            curr_degree += 1
    return (
        np.array(hall_set, dtype=int),
        np.array(degrees, dtype=int),
        np.array(degree_boundaries, dtype=int),
        reverse_map,
        width,
    )


@lru_cache(maxsize=0)
def hb_to_string(z, width, desired_degree):
    """
hb_to_string( 7 , 3, 6)
'[1,[1,2]]'

    """
    np_hall_set = hall_basis(width, desired_degree)[0]
    (n, m) = np_hall_set[z]
    if n:
        return (
            "["
            + hb_to_string(n, width, desired_degree)
            + ","
            + hb_to_string(m, width, desired_degree)
            + "]"
        )
    else:
        return str(m)


@lru_cache(maxsize=0)
def logsigkeys(width, desired_degree):
    """
logsigkeys(3,6)
' 1 2 3 [1,2] [1,3] [2,3] [1,[1,2]] [1,[1,3]] [2,[1,2]] [2,[1,3]] [2,[2,3]] [3,[1,2]] [3,[1,3]] [3,[2,3]] [1,[1,[1,2]]] [1,[1,[1,3]]] [2,[1,[1,2]]] [2,[1,[1,3]]] [2,[2,[1,2]]] [2,[2,[1,3]]] [2,[2,[2,3]]] [3,[1,[1,2]]] [3,[1,[1,3]]] [3,[2,[1,2]]] [3,[2,[1,3]]] [3,[2,[2,3]]] [3,[3,[1,2]]] [3,[3,[1,3]]] [3,[3,[2,3]]] [[1,2],[1,3]] [[1,2],[2,3]] [[1,3],[2,3]] [1,[1,[1,[1,2]]]] [1,[1,[1,[1,3]]]] [2,[1,[1,[1,2]]]] [2,[1,[1,[1,3]]]] [2,[2,[1,[1,2]]]] [2,[2,[1,[1,3]]]] [2,[2,[2,[1,2]]]] [2,[2,[2,[1,3]]]] [2,[2,[2,[2,3]]]] [3,[1,[1,[1,2]]]] [3,[1,[1,[1,3]]]] [3,[2,[1,[1,2]]]] [3,[2,[1,[1,3]]]] [3,[2,[2,[1,2]]]] [3,[2,[2,[1,3]]]] [3,[2,[2,[2,3]]]] [3,[3,[1,[1,2]]]] [3,[3,[1,[1,3]]]] [3,[3,[2,[1,2]]]] [3,[3,[2,[1,3]]]] [3,[3,[2,[2,3]]]] [3,[3,[3,[1,2]]]] [3,[3,[3,[1,3]]]] [3,[3,[3,[2,3]]]] [[1,2],[1,[1,2]]] [[1,2],[1,[1,3]]] [[1,2],[2,[1,2]]] [[1,2],[2,[1,3]]] [[1,2],[2,[2,3]]] [[1,2],[3,[1,2]]] [[1,2],[3,[1,3]]] [[1,2],[3,[2,3]]] [[1,3],[1,[1,2]]] [[1,3],[1,[1,3]]] [[1,3],[2,[1,2]]] [[1,3],[2,[1,3]]] [[1,3],[2,[2,3]]] [[1,3],[3,[1,2]]] [[1,3],[3,[1,3]]] [[1,3],[3,[2,3]]] [[2,3],[1,[1,2]]] [[2,3],[1,[1,3]]] [[2,3],[2,[1,2]]] [[2,3],[2,[1,3]]] [[2,3],[2,[2,3]]] [[2,3],[3,[1,2]]] [[2,3],[3,[1,3]]] [[2,3],[3,[2,3]]] [1,[1,[1,[1,[1,2]]]]] [1,[1,[1,[1,[1,3]]]]] [2,[1,[1,[1,[1,2]]]]] [2,[1,[1,[1,[1,3]]]]] [2,[2,[1,[1,[1,2]]]]] [2,[2,[1,[1,[1,3]]]]] [2,[2,[2,[1,[1,2]]]]] [2,[2,[2,[1,[1,3]]]]] [2,[2,[2,[2,[1,2]]]]] [2,[2,[2,[2,[1,3]]]]] [2,[2,[2,[2,[2,3]]]]] [3,[1,[1,[1,[1,2]]]]] [3,[1,[1,[1,[1,3]]]]] [3,[2,[1,[1,[1,2]]]]] [3,[2,[1,[1,[1,3]]]]] [3,[2,[2,[1,[1,2]]]]] [3,[2,[2,[1,[1,3]]]]] [3,[2,[2,[2,[1,2]]]]] [3,[2,[2,[2,[1,3]]]]] [3,[2,[2,[2,[2,3]]]]] [3,[3,[1,[1,[1,2]]]]] [3,[3,[1,[1,[1,3]]]]] [3,[3,[2,[1,[1,2]]]]] [3,[3,[2,[1,[1,3]]]]] [3,[3,[2,[2,[1,2]]]]] [3,[3,[2,[2,[1,3]]]]] [3,[3,[2,[2,[2,3]]]]] [3,[3,[3,[1,[1,2]]]]] [3,[3,[3,[1,[1,3]]]]] [3,[3,[3,[2,[1,2]]]]] [3,[3,[3,[2,[1,3]]]]] [3,[3,[3,[2,[2,3]]]]] [3,[3,[3,[3,[1,2]]]]] [3,[3,[3,[3,[1,3]]]]] [3,[3,[3,[3,[2,3]]]]] [[1,2],[1,[1,[1,2]]]] [[1,2],[1,[1,[1,3]]]] [[1,2],[2,[1,[1,2]]]] [[1,2],[2,[1,[1,3]]]] [[1,2],[2,[2,[1,2]]]] [[1,2],[2,[2,[1,3]]]] [[1,2],[2,[2,[2,3]]]] [[1,2],[3,[1,[1,2]]]] [[1,2],[3,[1,[1,3]]]] [[1,2],[3,[2,[1,2]]]] [[1,2],[3,[2,[1,3]]]] [[1,2],[3,[2,[2,3]]]] [[1,2],[3,[3,[1,2]]]] [[1,2],[3,[3,[1,3]]]] [[1,2],[3,[3,[2,3]]]] [[1,2],[[1,2],[1,3]]] [[1,2],[[1,2],[2,3]]] [[1,3],[1,[1,[1,2]]]] [[1,3],[1,[1,[1,3]]]] [[1,3],[2,[1,[1,2]]]] [[1,3],[2,[1,[1,3]]]] [[1,3],[2,[2,[1,2]]]] [[1,3],[2,[2,[1,3]]]] [[1,3],[2,[2,[2,3]]]] [[1,3],[3,[1,[1,2]]]] [[1,3],[3,[1,[1,3]]]] [[1,3],[3,[2,[1,2]]]] [[1,3],[3,[2,[1,3]]]] [[1,3],[3,[2,[2,3]]]] [[1,3],[3,[3,[1,2]]]] [[1,3],[3,[3,[1,3]]]] [[1,3],[3,[3,[2,3]]]] [[1,3],[[1,2],[1,3]]] [[1,3],[[1,2],[2,3]]] [[1,3],[[1,3],[2,3]]] [[2,3],[1,[1,[1,2]]]] [[2,3],[1,[1,[1,3]]]] [[2,3],[2,[1,[1,2]]]] [[2,3],[2,[1,[1,3]]]] [[2,3],[2,[2,[1,2]]]] [[2,3],[2,[2,[1,3]]]] [[2,3],[2,[2,[2,3]]]] [[2,3],[3,[1,[1,2]]]] [[2,3],[3,[1,[1,3]]]] [[2,3],[3,[2,[1,2]]]] [[2,3],[3,[2,[1,3]]]] [[2,3],[3,[2,[2,3]]]] [[2,3],[3,[3,[1,2]]]] [[2,3],[3,[3,[1,3]]]] [[2,3],[3,[3,[2,3]]]] [[2,3],[[1,2],[1,3]]] [[2,3],[[1,2],[2,3]]] [[2,3],[[1,3],[2,3]]] [[1,[1,2]],[1,[1,3]]] [[1,[1,2]],[2,[1,2]]] [[1,[1,2]],[2,[1,3]]] [[1,[1,2]],[2,[2,3]]] [[1,[1,2]],[3,[1,2]]] [[1,[1,2]],[3,[1,3]]] [[1,[1,2]],[3,[2,3]]] [[1,[1,3]],[2,[1,2]]] [[1,[1,3]],[2,[1,3]]] [[1,[1,3]],[2,[2,3]]] [[1,[1,3]],[3,[1,2]]] [[1,[1,3]],[3,[1,3]]] [[1,[1,3]],[3,[2,3]]] [[2,[1,2]],[2,[1,3]]] [[2,[1,2]],[2,[2,3]]] [[2,[1,2]],[3,[1,2]]] [[2,[1,2]],[3,[1,3]]] [[2,[1,2]],[3,[2,3]]] [[2,[1,3]],[2,[2,3]]] [[2,[1,3]],[3,[1,2]]] [[2,[1,3]],[3,[1,3]]] [[2,[1,3]],[3,[2,3]]] [[2,[2,3]],[3,[1,2]]] [[2,[2,3]],[3,[1,3]]] [[2,[2,3]],[3,[2,3]]] [[3,[1,2]],[3,[1,3]]] [[3,[1,2]],[3,[2,3]]] [[3,[1,3]],[3,[2,3]]]'

    """
    np_hall_set, np_degrees, np_degree_boundaries, reverse_map, width = hall_basis(
        width, desired_degree
    )
    return " " + " ".join(
        [hb_to_string(z, width, desired_degree) for z in range(1, np_hall_set.shape[0])]
    )


def lie_to_string(li, width, depth):
    """
lie_to_string(prod(7, 6, 3, 6), 3, 6)
'-1.0 [[2,3],[1,[1,2]]]'

    """
    return " + ".join(
        [str(li[x]) + " " + hb_to_string(x, width, depth) for x in sorted(li.keys())]
    )


def key_to_sparse(k):
    """
>>> key_to_sparse(7)
defaultdict(<class 'float'>, {7: 1.0})
>>> 
    """
    ans = defaultdict(scalar_type)
    ans[k] = scalar_type(1)
    return ans


## add and subtract and scale sparse scalar_type vectors based on defaultdict class
"""
>>> lhs = key_to_sparse(3)
>>> rhs = key_to_sparse(5)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> subtract_into(lhs, rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 0.0})
>>> 
"""


def add_into(lhs, rhs):
    for k in rhs.keys():
        lhs[k] += rhs.get(k, scalar_type())
    return lhs


def subtract_into(lhs, rhs):
    for k in rhs.keys():
        lhs[k] -= rhs.get(k, scalar_type())
    return lhs


def scale_into(lhs, s):
    """
>>> rhs = key_to_sparse(5)
>>> lhs = key_to_sparse(3)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> scale_into(lhs, 3)
defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
>>> scale_into(lhs, 0)
defaultdict(<class 'float'>, {})
>>> 
    """
    if s:
        for k in lhs.keys():
            lhs[k] *= s
    else:
        lhs = defaultdict(scalar_type)
    return lhs


def sparsify(arg):
    """
>>> rhs = key_to_sparse(5)
>>> lhs = key_to_sparse(3)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> subtract_into(lhs, rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 0.0})
>>> sparsify(lhs)
defaultdict(<class 'float'>, {3: 1.0})
>>> 
    """
    empty_key_vals = list(k for k in arg.keys() if not arg[k])
    # an iterable would break
    for k in empty_key_vals:
        del arg[k]
    return arg


def multiply(lhs, rhs, func):
    ## WARNING assumes all multiplications are in range -
    ## if not then the product should use the coproduct and the max degree
    ans = defaultdict(scalar_type)
    for k1 in sorted(lhs.keys()):
        for k2 in sorted(rhs.keys()):
            add_into(ans, scale_into(func(k1, k2), lhs[k1] * rhs[k2]))
    return sparsify(ans)


@lru_cache(maxsize=0)
def prod(k1, k2, width, depth):
    """
>>> rhs = key_to_sparse(5)
>>> lhs = key_to_sparse(3)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> scale_into(lhs, 3)
defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
>>> subtract_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 3.0, 5: 2.0})
>>> _prod = lambda kk1, kk2: prod(kk1, kk2, 3, 6)
>>> multiply(lhs,rhs,_prod)
defaultdict(<class 'float'>, {13: 3.0})
>>> multiply(rhs,lhs,_prod)
defaultdict(<class 'float'>, {13: -3.0})
>>> multiply(lhs,lhs,_prod)
defaultdict(<class 'float'>, {})
>>> 
    """
    _prod = lambda kk1, kk2: prod(kk1, kk2, width, depth)
    if k1 > k2:
        ans = _prod(k2, k1)
        scale_into(ans, -scalar_type(1))
        return ans
    ans = defaultdict(scalar_type)
    hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(width, depth)
    if k1 == k2:
        return ans
    if degrees[k1] + degrees[k2] > depth:
        return ans
    t = reverse_map.get((k1, k2), 0)
    if t:
        ans[t] = scalar_type(1)
    else:
        (k3, k4) = hall_set[k2]  ## (np.int32,np.int32)
        k3 = int(k3)
        k4 = int(k4)
        ### We use Jacobi: [k1,k2] = [k1,[k3,k4]]] = [[k1,k3],k4]-[[k1,k4],k3]
        wk13 = _prod(k1, k3)
        wk4 = key_to_sparse(k4)
        t1 = multiply(wk13, wk4, _prod)
        t2 = multiply(_prod(k1, k4), key_to_sparse(k3), _prod)
        ans = subtract_into(t1, t2)
    return ans


def sparse_to_dense(sparse, width, depth):
    """
>>> rhs = key_to_sparse(5)
>>> lhs = key_to_sparse(3)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> scale_into(lhs, 3)
defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
>>> subtract_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 3.0, 5: 2.0})
>>> _prod = lambda kk1, kk2: prod(kk1, kk2, 3, 6)
>>> add_into(lhs , multiply(rhs,lhs,_prod))
defaultdict(<class 'float'>, {3: 3.0, 5: 2.0, 13: -3.0})
>>> sparse_to_dense(lhs,3,2)
array([0., 0., 3., 0., 2., 0.])
>>> 
    """
    ### is that last line correct??
    hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(width, depth)
    dense = np.zeros(len(hall_set), dtype=np.float64)
    for k in sparse.keys():
        if k < len(hall_set):
            dense[k] = sparse[k]
    return dense[1:]


def dense_to_sparse(dense, width, depth):
    """
>>> hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(2, 3)
>>> l = np.array( [i for i in range(1,len(hall_set))], dtype=np.float64)
>>> print(l," ",dense_to_sparse(l,2,3))
[1. 2. 3. 4. 5.]   defaultdict(<class 'float'>, {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0})
>>> hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(2, 3)
>>> l = np.array( [i for i in range(1,len(hall_set))], dtype=np.float64)
>>> print(l," ",dense_to_sparse(l,2,3))
[1. 2. 3. 4. 5.]   defaultdict(<class 'float'>, {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0})
>>> 
>>> sparse_to_dense(dense_to_sparse(l,2,3), 2,3) == l
array([ True,  True,  True,  True,  True])
>>> 
    """
    sparse = defaultdict(scalar_type)
    for k in range(len(dense)):
        if dense[k]:
            sparse[k + 1] = dense[k]
    return sparse


## expand is a map from hall basis keys to tensors
@lru_cache(maxsize=0)
def expand(k, width, depth):
    _expand = lambda k: expand(k, width, depth)
    _tensor_multiply = lambda k1, k2: tjl_dense_numpy_tensor.tensor_multiply(
        k1, k2, depth
    )
    _tensor_sub = tjl_dense_numpy_tensor.tensor_sub
    if k:
        hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(
            width, depth
        )
        (k1, k2) = hall_set[k]
        if k1:
            return _tensor_sub(
                _tensor_multiply(_expand(k1), _expand(k2)),
                _tensor_multiply(_expand(k2), _expand(k1)),
            )
        else:
            ans = tjl_dense_numpy_tensor.zero(width, 1)
            ans[blob_size(width, 0) - 1 + k2] = scalar_type(
                1
            )  ## recall k2 will never be zero
            return ans
    return tjl_dense_numpy_tensor.zero(width)


def l2t(arg, width, depth):
    """
>>> from tjl_hall_numpy_lie import *
>>> from tjl_dense_numpy_tensor import *
>>> width = 2
>>> depth = 3
>>> t = tensor_log(stream2sigtensor(brownian(100, width), depth), depth)
>>> print(np.sum(tensor_sub(l2t(t2l(t),width,depth), t)[2:]**2)  < 1e-30)
True
>>> 
    """
    _expand = lambda k: expand(k, width, depth)
    _tensor_add = tjl_dense_numpy_tensor.tensor_add
    ans = tjl_dense_numpy_tensor.zero(width)
    for k in arg.keys():
        if k:
            ans = tjl_dense_numpy_tensor.tensor_add(
                ans, (tjl_dense_numpy_tensor.rescale(_expand(k), arg[k]))
            )
    return ans


## tuple a1,a2,...,an is converted into [a1,[a2,[...,an]]] as a LIE element recursively.
@lru_cache(maxsize=0)
def rbraketing(tk, width, depth):
    _rbracketing = lambda t: rbraketing(t, width, depth)
    _prod = lambda x, y: prod(x, y, width, depth)
    _multiply = lambda k1, k2: multiply(k1, k2, _prod)
    hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(width, depth)
    if tk[1:]:
        return _multiply(_rbracketing(tk[:1]), _rbracketing(tk[1:]))
    else:
        ans = defaultdict(scalar_type)
        if tk:
            ans[tk[0]] = scalar_type(1)
        return ans


@lru_cache(maxsize=0)
def index_to_tuple(i, width):
    # the shape of the tensor that contains
    # \sum t[k] index_to_tuple(k,width) is the tensor
    # None () (0) (1) ...(w-1) (0,0) ...(n1,...,nd) ...
    """
>>> from tjl_hall_numpy_lie import *
>>> from tjl_dense_numpy_tensor import *
>>> width = 2
>>> depth = 3
>>> t = arange(width, depth)
>>> print(t)
[ 2.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15.]
>>> for t1 in [t]:
...     for k, coeff in enumerate(t1):
...         print (coeff, index_to_tuple(k,width))
... 
2.0 None
1.0 ()
2.0 (1,)
3.0 (2,)
4.0 (1, 1)
5.0 (1, 2)
6.0 (2, 1)
7.0 (2, 2)
8.0 (1, 1, 1)
9.0 (1, 1, 2)
10.0 (1, 2, 1)
11.0 (1, 2, 2)
12.0 (2, 1, 1)
13.0 (2, 1, 2)
14.0 (2, 2, 1)
15.0 (2, 2, 2)
>>> 
    """

    _blob_size = lambda depth: tjl_dense_numpy_tensor.blob_size(width, depth)
    _layers = lambda bz: tjl_dense_numpy_tensor.layers(bz, width)
    bz = i + 1
    d = _layers(bz)  ## this index is in the d-tensors
    if _layers(bz) < 0:
        return
    j = bz - 1 - _blob_size(d - 1)
    ans = ()
    ## remove initial offset to compute the index
    if j >= 0:
        for jj in range(d):
            ans = (1 + (j % width),) + ans
            j = j // width
    return ans


def t_to_string(i, width):
    j = index_to_tuple(i, width)
    if index_to_tuple(i, width) == None:
        return " "
    return "(" + ",".join([str(k) for k in index_to_tuple(i, width)]) + ")"


def sigkeys(width, desired_degree):
    """
>>> from tjl_hall_numpy_lie import *
>>> from tjl_dense_numpy_tensor import *
>>> width = 2
>>> depth = 3
>>> from esig import tosig as ts
>>> ts.sigkeys(width , depth) == sigkeys(width , depth)
True
>>> sigkeys(width , depth)
' () (1) (2) (1,1) (1,2) (2,1) (2,2) (1,1,1) (1,1,2) (1,2,1) (1,2,2) (2,1,1) (2,1,2) (2,2,1) (2,2,2)'
>>> 
    """
    t_to_string(0, width)
    return " " + " ".join(
        [t_to_string(z, width) for z in range(1, blob_size(width, desired_degree))]
    )


def t2l(arg):
    # projects a lie element in tensor form to a lie in lie basis form
    """
>>> from tjl_hall_numpy_lie import *
>>> from tjl_dense_numpy_tensor import *
>>> width = 2
>>> depth = 3
>>> t = tensor_log(stream2sigtensor(brownian(100, width), depth), depth)
>>> print(np.sum(tensor_sub(l2t(t2l(t),width,depth), t)[2:]**2)  < 1e-30)
True
>>> 
    """
    width = int(arg[0])
    _layers = lambda bz: tjl_dense_numpy_tensor.layers(bz, width)
    _blob_size = lambda dep: tjl_dense_numpy_tensor.blob_size(width, dep)
    depth = _layers(len(arg))
    ans = defaultdict(scalar_type)
    ibe = _blob_size(0)  # just beyond the zero tensors
    ien = _blob_size(depth)
    for i in range(ibe, ien):
        t = index_to_tuple(i, width)
        if t:
            ## must normalise to get the dynkin projection to make it a projection
            add_into(
                ans,
                scale_into(rbraketing(t, width, depth), arg[i] / scalar_type(len(t))),
            )
    return ans


"""

	inline TENSOR l2t(const LIE& arg)
	{
		TENSOR result;
		typename LIE::const_iterator i;
		for (i = arg.begin(); i != arg.end(); ++i)
			result.add_scal_prod(expand(i->first), i->second);
		return result;
	}
	/// Returns the free lie element corresponding to a tensor_element.
	/**
	This is the Dynkin map obtained by right bracketing. Of course, the
	result makes sense only if the given free_tensor is the tensor expression
	of some free lie element.
	*/
	inline LIE t2l(const TENSOR& arg)
	{
		LIE result;
		typename TENSOR::const_iterator i;
		for (i = arg.begin(); i != arg.end(); ++i)
			result.add_scal_prod(rbraketing(i->first), i->second);
		typename LIE::iterator j;
		for (j = result.begin(); j != result.end(); ++j)
			(j->second) /= (RAT)(LIE::basis.degree(j->first));
		return result;
	}
	/// For a1,a2,...,an, return the expression [a1,[a2,[...,an]]].
	/**
	For performance reasons, the already computed expressions are stored in a
	static table to speed up further calculus. The function returns a
	constant reference to an element of this table.
	*/
	inline const LIE& rbraketing(const TKEY& k)
	{
		static boost::recursive_mutex table_access;
		// get exclusive recursive access for the thread
		boost::lock_guard<boost::recursive_mutex> lock(table_access);

		static std::map<TKEY, LIE> lies;
		typename std::map<TKEY, LIE>::iterator it;
		it = lies.find(k);
		if (it == lies.end())
			return lies[k] = _rbraketing(k);
		else
			return it->second;
	}
	/// Returns the free_tensor corresponding to the Lie key k.
	/**
	For performance reasons, the already computed expressions are stored in a
	static table to speed up further calculus. The function returns a
	constant reference to an element of this table.
	*/
	inline const TENSOR& expand(const LKEY& k)
	{
		static boost::recursive_mutex table_access;
		// get exclusive recursive access for the thread
		boost::lock_guard<boost::recursive_mutex> lock(table_access);

		static std::map<LKEY, TENSOR> table;
		typename std::map<LKEY, TENSOR>::iterator it;
		it = table.find(k);
		if (it == table.end())
			return table[k] = _expand(k);
		else
			return it->second;
	}
private:
	/// Computes recursively the free_tensor corresponding to the Lie key k.
	TENSOR _expand(const LKEY& k)
	{
		if (LIE::basis.letter(k))
			return (TENSOR)TENSOR::basis.keyofletter(LIE::basis.getletter(k));
		return commutator(expand(LIE::basis.lparent(k)),
			expand(LIE::basis.rparent(k)));
	}
	/// a1,a2,...,an is converted into [a1,[a2,[...,an]]] recursively.
	LIE _rbraketing(const TKEY& k)
	{
		if (TENSOR::basis.letter(k))
			return (LIE)LIE::basis.keyofletter(TENSOR::basis.getletter(k));
		return rbraketing(TENSOR::basis.lparent(k))
			* rbraketing(TENSOR::basis.rparent(k));
	}
};
"""

"""
/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurko and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */




//  lie_basis.h


// Include once wrapper
#ifndef DJC_COROPA_LIBALGEBRA_LIEBASISH_SEEN
#define DJC_COROPA_LIBALGEBRA_LIEBASISH_SEEN

/// The Hall Basis class.
/**

   A basis is a finite total ordered set of keys, its cardinal is size() and
   its minimal element is begin(). The successor key of a given key is given
   by nextkey(). The successor of the maximal key is end() and does not belong
   to the basis. The position of a given key in the total order of the basis
   is given by keypos(), and equals 1 for begin(). To each letter corresponds
   a key.

   This class is an ancestor of the lie_basis class, used to implement the lie
   class (Free Lie Algebra) as a particular instance of an algebra class
   (Associative Algebras).

   This class stores a Philip Hall basis associated to a finite number of
   letters. A key is the implementation of a Lie element of this basis. A
   letter is a particular Lie element (or basis element, or key). Each key k
   which does not correspond to a letter has two parents lp and rp and we have
   k = [lp,rp] where [.,.] is the Lie product. A letter, viewed as a key, has
   no parents. More precisely, its parents are invalid keys. 

   The basis elements are recursively computed and are enumerated with keys.
   The set of valid keys is essentially an interval of natural integers.
   
   One can find below a brief Mathematical description of Philip Hall bases
   for the free Lie Algebra. Cf. Reutenauer's book for example, ISBN 0 19
   853679 8.
   
   Let K be a field with characteristic non equals to 2. In
   newgenesis-libalgebra, this field K corresponds to the type SCA defined in
   libalgebra.h.
   
   Let M be a finite alphabet {a_1,...,a_n}. We denote by M* the monoid which
   consists in words of letters in M. The product in M* is the concatenation
   and the neutral element is the empty word.
   
   We consider the free albegra A over (K,M). An element of A is a linear
   combination of elements of M*, with coefficients in K. An element of A is
   an instance of class free_tensor<>, which affects to each element of M* a
   coefficient in K. The element of M* are indexed by tensor_key<>, which
   essentially stores the corresponding word as a std::string.
   
   We consider also the associated free Lie albegra L, the smallest subalgebra
   of A which contains M and is stable by the Lie product [X,Y] = XY-YX. An
   element of L is an instance of class lie<>. The key used are of type
   lie_key<>, which are actually indexes in a basis of type lie_basis<>.
   
   The degree of a word w in M is its length. The degree of an element of the
   algebra A is the maximum degree of words with non zero coefficients. The
   degree of [X,Y] is the sum of the degrees of X and Y if X and Y are
   different, and 0 if X = Y.
   
   Actually, the free Lie algebra L is a graded algebra, with respect to the
   degree (or weight) of Lie products. Philip Hall invented an algorithm for
   computing a basis of the free Lie albegra L. A Hall basis H is a union of
   subsets H_1,...H_i,... of L. By definition, H_1 = M = {a_1,...,a_n} and the
   elements of H_i are of degree i. The set H is totally ordered and more
   over, H_1 < H_2 < ... The Hall basis H can be constructed recursively from
   H_1. This can be done by constructing an array HALLARRAY of elements of the
   form {left, degree , right}. The left and right corresponds to indexes in
   the array for constructing the element by the Lie product, and degree
   corresponds to the degree of this element, which is then the sum of the
   degrees of the two elements pointed by left and right. The order of the
   elements of the array is in one to one correspondance with the order of H.
   The subset H_i is exactly the elements of the form {left, degree , right}
   with degree = i.
   
   Starting from H1 = {{0, 1, 1},...,{0, 1, n}} which corresponds to the n
   letters, Hi+1 is constructed from H_1, ..., H_i by examining all elements
   of the form {l, i + 1, r} where l < r and l and r are in the union of
   H_1,...,H_i. Such an element is added to the set Hi+1 if and only if the
   right parent of r is <= l.
*/
// Hall basis provides a fully populated (i.e. dense) basis for the lie elements
// has the ability to extend the basis by degrees. This is not protected or thread safe.
// To avoid error in multi-threaded code it is essential that it is extended at safe times (e.g.
// on construction). The code for lie basis has been modified to do this and avoid a subtle error.

// It would be worthwhile to write a data driven sparse hall basis

class hall_basis
{
public:
	/// The default key has value 0, which is an invalid value
	/// and occurs as a parent key of any letter.
	/// 
	/// keys can get large - but in the dense case this is not likely
	/// Make a choice for the length of a key in 64 bit.
	typedef DEG KEY; // unsigned int
	//typedef LET KEY; // unsigned longlong
	/// The parents of a key are a pair of prior keys. Invalid 0 keys for letters.
	typedef std::pair<KEY, KEY> PARENT;
protected:
	/// Parents, indexed by keys.
	std::vector<PARENT> hall_set;
	/// Reverse map from parents to keys.
	std::map<PARENT, KEY> reverse_map;
	/// Degrees, indexed by keys.
	std::vector<DEG> degrees;
	/// Letters, indexed by their keys.
	//std::string letters;
	std::vector<LET> letters;
	/// Maps letters to keys.
	std::map<LET, KEY> ltk;
	/// Current degree, always > 0 for any valid class instance.
	DEG curr_degree;
public:
	/// Constructs the basis with a given number of letters.
	hall_basis(DEG n_letters)
		: curr_degree(0)
	{
		// We put something at position 0 to make indexing more logical
		degrees.push_back(0);
		PARENT p(0,0);
		hall_set.push_back(p);
	
		for (LET c = 1; c <= n_letters; ++c)
			letters.push_back(c); //+= (char) c;
	
		// We add the letters to the basis, starting from position 1.
		KEY i;
		for (i = 1; i <= letters.size(); ++i)
		{
			PARENT parents(0,i);
			hall_set.push_back(parents); // at [i]
			degrees.push_back(1); // at [i]
			ltk[letters[i - 1]] = (LET) i;
		}
		curr_degree = 1;
		// To construct the rest of the basis now, call growup(max_degree) here.
	}
	/// Constructs the basis up to a desired degree. 
	/**
	For performance reasons, max_degree is not checked. So be careful.
	*/
	inline void growup(DEG desired_degree)
	{
		for (DEG d = curr_degree + 1; d <= desired_degree; ++d)
		{
			KEY bound = (KEY)len(hall_set);
			for (KEY i = 1; i <= bound; ++i)
				for (KEY j = i + 1; j <= bound; ++j)
					if ((degrees[i] + degrees[j] == d) && (hall_set[j].first <= i))
					{
						PARENT parents(i, j);
						hall_set.push_back(parents);  // at position max_key.
						degrees.push_back(d);         // at position max_key.
						reverse_map[parents] = (KEY) len(hall_set) - 1;
					}
			++curr_degree;
		}
	}
	/// Returns the degree (ie. weight) of a Lie key.
	inline DEG degree(const KEY& k) const
	{
		return degrees[k];
	}
	/// Returns the key corresponding to a letter.
	inline KEY keyofletter(LET letter) const
	{
		return ltk.find(letter)->second;
	}
	/// Returns the left parent of a key. 
	inline KEY lparent(const KEY& k) const
	{
		return hall_set[k].first;
	}
	/// Returns the right parent of a key.
	inline KEY rparent(const KEY& k) const
	{
		return hall_set[k].second;
	}
	/// Tells if a key corresponds to a letter.
	inline bool letter(const KEY& k) const
	{
		return ((k > 0) && (k <= letters.size()));
	}
	/// Returns the letter of a key corresponding to a letter.
	inline LET getletter(const KEY& k) const
	{
		return letters[k - 1];
	}
	/// Returns the value of the smallest key in the basis.
	inline KEY begin(void) const
	{
		return 1;
	}
	/// Returns the key next the biggest key of the basis.
	inline KEY end(void) const
	{
		return 0;
	}
	/// Returns the key next a given key in the basis. No implicit growup made.
	inline KEY nextkey(const KEY& k) const
	{
		if (k < (len(hall_set) - 1))
			return (k + 1);
		else
			return 0;
	}
	/// Returns the position of a key in the basis total order.
	inline DEG keypos(const KEY& k) const
	{
		return k;
	}
	/// Returns the size of the basis.
	inline DEG size(void) const
	{
		return ( (KEY) len(hall_set) - 1);
	}
	/// Outputs the Hall basis to an std::ostream.
	inline friend std::ostream& operator<<(std::ostream& os, hall_basis& b)
	{	
		for (KEY k = b.begin(); k != b.end(); k = b.nextkey(k))
			os << b.key2string(k) << ' ';
		return os;
	}

	//inline const std::string& key2string(const KEY& k) const
	//BUG//TJL//24/08/2012 - returned reference invalidated if vector grows!!
	//BUG//TJL//25/08/2012 - not templated but has static member so this is shared across all dimensions regardless of no letters etc!!
	private:
		mutable std::vector<std::string> table; //move to instance per class
	public:

	//ShortFix return a value not a reference
	//TODO check performance of fix 24/08/2012
	/// Converts a key to an std::string of letters.
	
	inline const std::string key2string(const KEY& k) const
	{
		static boost::recursive_mutex table_access;
		//// get exclusive recursive access for the thread 
		boost::lock_guard<boost::recursive_mutex> lock(table_access); 

		//BUG//TJL//09/04/2017 - non static member added to class but not commented out here!!
//		static std::vector<std::string> table;

		if (k > table.size())
		{
			for (KEY i = (KEY) table.size() + 1; i <= k; ++i)
				table.push_back(_key2string(i));
		}
		return table[k - 1];
	}
private:
	/// Recursively constructs the string associated to the Lie key k.
	std::string _key2string(const KEY& k) const
	{
		std::ostringstream oss;
		if (k > 0)
		{
			if (letter(k))
				oss << getletter(k);
			else
			{
				oss << '[';
				oss << key2string(lparent(k));
				oss << ',';
				oss << key2string(rparent(k));
				oss << ']';
			}
		}
		return oss.str();
	}
};

/// The Lie basis class.
/** 
 This is the basis used to implement the lie class as a specialisation of
 the algebra class. In the current implementation, the Lie basis class is a
 wrapper for the hall_basis class, with a prod() member function.
*/

template<typename SCA, typename RAT, DEG n_letters, DEG max_degree>
class lie_basis : public hall_basis,
				  public basis_traits<With_Degree, n_letters, max_degree>
{
public:
	/// Import of the KEY type.
	typedef hall_basis::KEY KEY;
	/// The MAP type.
	typedef std::map<KEY, SCA> MAP;
	/// The Free Lie Associative Algebra element type.
	typedef lie<SCA, RAT, n_letters, max_degree> LIE;
	/// The rationals.
	typedef RAT RATIONAL;
public:
	/// Constructs the basis for a finite number of letters.
	lie_basis(void)
		: hall_basis(n_letters) {
		// bug: tjl : 08 04 2017 without the following line the basis would not remain const and sharing it between threads would cause errors
		hall_basis::growup(max_degree);
	}
	/// Returns the product of two key.
	/**
	Returns the LIE instance corresponding to the product of the two basis
	elements k1 and k2. For performance reasons, the basis is enriched/grown
	dynamically and the already computed products are stored in a static
	multiplication table to speed up further calculations. This function
	returns a constant reference to the suitable table element.
	*/
	inline const LIE& prod(const KEY& k1, const KEY& k2)
	{
		static boost::recursive_mutex table_access;
		// get exclusive recursive access for the thread 
		boost::lock_guard<boost::recursive_mutex> lock(table_access); 

		static std::map<PARENT, LIE> table;
		static typename std::map<PARENT, LIE>::iterator it;
		PARENT p(k1, k2);
		it = table.find(p);
		if (it == table.end())
			return table[p] = _prod(k1, k2);
		else
			return it->second;
	}
	/// Replaces letters by lie<> instances in a lie<> instance.
	/**
	Replaces the occurences of s letters in the expression of k by the lie<>
	elements in v, and returns the recursively expanded result. The already
	computed replacements are stored in table.
	*/
	LIE replace(const KEY& k, const std::vector<LET>& s, const std::vector<LIE*>& v, std::map<KEY, LIE>& table)
	{
		typename std::map<KEY, LIE>::iterator it;
		it = table.find(k);
		if (it != table.end())
			return it->second;
		else
		{
			if (letter(k))
			{
				typename std::vector<LET>::size_type i;
				for (i = 0; i < s.size(); ++i)
					if (s[i] == getletter(k))
						return table[k] = *(v[i]);
				return (table[k] = (LIE)k);
			}
			else
				return (table[k]
						= replace(lparent(k), s, v, table)
						* replace(rparent(k), s, v, table));
		}
	}
private:
	/// The recursive key product.
	LIE _prod(const KEY& k1, const KEY& k2)
	{ 	
		LIE empty;
		// [A,A] = 0.
		if (k1 == k2)
			return empty;
		// if index(A) > index(B) we use [A,B] = -[B,A] 
		if (k1 > k2)
			return -prod(k2, k1);
		//
		DEG target_degree = degrees[k1] + degrees[k2];
		if ((max_degree > 0) && (target_degree > max_degree))
			return empty; // degree truncation
		// We grow up the basis up to the desired degree.
		growup(target_degree);
		// We look up for the desired product in our basis.
		PARENT parents(k1, k2);
		typename std::map<PARENT, KEY>::const_iterator it;
		it = reverse_map.find(parents);
		if (it != reverse_map.end())
		{
			// [k1,k2] exists in the basis.
			LIE result(it->second);
			return result;
		}
		else
			// [k1,k2] does not exists in the basis.
		{
			// Since k1 <= k2, k2 is not a letter because if it was a letter, 
			// then also k1, which is impossible since [k1,k2] is not in the basis.
			// We use Jacobi: [k1,k2] = [k1,[k3,k4]]] = [[k1,k3],k4]-[[k1,k4],k3] 
			KEY k3(lparent (k2));
			KEY k4(rparent (k2));
			LIE result(prod(k1, k3) * (LIE)k4);
			return result.sub_mul(prod(k1, k4), (LIE)k3);
		}
	}
	/// Outupts an std::pair<lie_basis*, KEY> to an std::ostream.
	inline friend std::ostream& operator<<(std::ostream& os, const std::pair<lie_basis*, KEY>& t)
	{
		return os << t.first->key2string(t.second);
	}
};


// Include once wrapper
#endif // DJC_COROPA_LIBALGEBRA_LIEBASISH_SEEN

//EOF.

"""

if __name__ == "__main__":
    ##np.set_printoptions(suppress=True,formatter={'float_kind':'{:16.3f}'.format},
    ##linewidth=130)
    # string with the expression # np.array_repr(x, precision=6,
    # suppress_small=True)
    doctest.testmod()
