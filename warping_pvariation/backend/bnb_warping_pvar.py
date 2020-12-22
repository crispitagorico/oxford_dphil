import sys
sys.path.insert(0, '../data')

import pybnb
import numpy as np
import math
import copy
import time
import matplotlib.pyplot as plt
from functools import lru_cache
import pylru
from collections import defaultdict
import iisignature

import pvar_backend
import pvar_tools 

def iisignature_sig(x, d):
    if len(x)<=1:
        return np.zeros(iisignature.siglength(x.shape[1],d))
    else:
        return iisignature.sig(x,d)

class BnBWarping(pybnb.Problem):

    
    """ ##########################################################################################################
    
            Fast version of BnB Warping P-variation distance algorithm:

            Let X: I ----> R^d & Y: J ----> R^d be two un-parameterized (rough) paths.
            Let a: K ---> I, b: K ---> J be two increasing, surjective functions, 
                        with K being a commong parameterization.

            We are interested in computing the "warping p-variation" distance between 2 rough paths
            X and Y. This metric is defined in terms of standard p-variation distance d_pvar
            (equivalent to the Carnot-Caratheodory metric)

                                inf_{a, b} d_pvar(X o a, Y o b)      (1)

            where: - p is the p-ariation hyper-parameter
                   - o means composition of functions

            Given a stream x on R^d, we can extend it to a rough path X via the signature transform.
            In this code we use the library signatory (developed by Patrick Kidger, ref ........)
            to compute and query path signatures.

            To minimize over all possible parameterization a, b we developed a Branch-and-Bound
            algorithm explained in the paper ............ We make use of the python library "pybnb"
            (ref .......). The solver in pybnb keeps track of the best solution seen so far for you, 
            and will prune the search space by calling the branch() method when it 
            encounters a node whose bound() is worse than the best objective() seen so far.

            The problem can be simplified if the we consider bijective parameterizations a, b.
            In such case, the space of parameterizations is a a group under composition o.
            Therefore any tuple (a,b) can be re-written as (id, b o a^-1). 
            Hence the minimization reduces to the following

                                    inf_{b} d_pvar(X, Y o b)             (2)

            In the code below, setting the attribute "simplied = True" solves the second minimization (2).
          
                
        ############################################################################################################   
    
    
    
                The solver in pybnb keeps track of the best solution seen so far for you, 
                    and will prune the search space by calling the branch() method when it 
                    encounters a node whose bound() is worse than the best objective() seen so far.

                    ##################################################################################
                        
                            This is the version to use for experiments and visualizations ONLY!
                            It also allows for visualizing results and computing stats
                                    (and use memoization, sequentially)

                    ##################################################################################

    """

    def __init__(self, x, x_inv, y, y_inv, p, norm='l1', root_node=(0,0), bc=4, width=2,
                 pvar_dist_mem=None, cache_size=2**10, rough_path=False,
                 
                 pvar_mem_org=None,
                 plot_2d=True,
                 record_path=False, 
                 initial_time=None,
                 use_bound1=True, 
                 use_bound2=True, 
                 use_bound3=True, 
                 ):

        """Inputs:
                   - x, y: input paths
                   - x_inv, y_inv: their inverses in the group
                   - p: p for p-variation
                   - depth: signature truncation depth
                   - norm: norm for pairwise signature distance 
                   - root_node: BnB root node at the start of the tree
                   - bc: boundary conditon for starting using the tight bound after bc steps in the tree
                   - record_path: whether to store nodes and edges of the tree (use for visualization only)
                   - plot_2d: whether to plot results of 1-d or 2-d paths (use for visualization only)
                   - pvar_dist_mem: memoization dictionary for p-var distances
                   - pvar_mem_org: memoization dictionary to monitor distribution of pvar_dist_mem (use for visualization only)
                   - initial_time: starting time of the procedure (use for testing only)
                   - use_boundi: flag that determines whether to use or not bound i, for all i in {1,2,3}"""

        # input paths
        lx = x.shape[0]
        ly = y.shape[0]
        if ly >= lx:
            self.x = y
            self.x_inv = y_inv
            self.y = x
            self.y_inv = x_inv
        else:
            self.x = x
            self.x_inv = x_inv
            self.y = y
            self.y_inv = y_inv

        self.m = len(self.x)
        self.n = len(self.y)
        assert self.m > 0
        assert self.n > 0

        self.p = p # p for p-variation
        self.depth = int(p) # signature depth
        self.width = width
        self.norm = norm # l1 or l2 norm

        self.rough_path = rough_path

        self.path = [(0,0), (0,0)] # record lattice path
        self.best_node_value = math.inf # keep track of best bound
        self.i0, self.j0 = root_node # tuple of indeces of root node

        self.plot_2d = plot_2d # flag to plot paths in 1-d or 2-d case
        self.cache_size = cache_size # size of the cache to store p-var and bound values

        # boundary condition, i.e. necessary min # of steps from root node to start using tight bound
        self.bc = bc 

        # This is a list that records the total size of the p-var cache as the algorithm runs (use for visualization only)
        self.total_size_of_pvar_cache = defaultdict(int)

        # pvar internal pathlets memoization
        if pvar_dist_mem is None:

            self.pvar_dist_mem = pylru.lrucache(self.cache_size)
            
            # (for visualization only)
            # This dictionary is for monitoring the structure of the memoization dictionary.
            # It basically consists of keys representing lattice warps. At each key is 
            # associated a list recording every time at which the key is hit during memoization.
            self.pvar_mem_org = defaultdict(list)

        else: # feed to the class called recursively the current state of the mem dictionary
            self.pvar_dist_mem = pvar_dist_mem
            self.pvar_mem_org = pvar_mem_org

        # here we store all nodes and all edges of the tree that is being built (use for visualization only)
        self.record_path = record_path
        if self.record_path:
            self.nodes = []
            self.edges = [] 

        ## here we use flags to specify what kind of bounds we want to use
        self.use_bound1 = use_bound1
        self.use_bound2 = use_bound2
        self.use_bound3 = use_bound3

        ## starting time of procedure (use for visualization only)
        if initial_time is None: 
            self.initial_time = time.time()
        else:
            self.initial_time = initial_time

    def simple_complexity(self, p, q):  
        """Returns number of paths from the southwest corner (0, 0) of a rectangular 
           grid to the northeast corner (p, q), using only single steps north or east
        """
        dp = [1 for i in range(q)] 
        for i in range(p - 1): 
            for j in range(1, q): 
                dp[j] += dp[j - 1] 
        return dp[q - 1]
    

    def Delannoy_number(self, m, n):
        """Returns number number of paths from the southwest corner (0, 0) of a rectangular 
           grid to the northeast corner (m, n), using only single steps north, northeast, or east
        """
        if (m == 0 or n == 0): 
            return 1
        return self.Delannoy_number(m-1, n) + self.Delannoy_number(m-1, n-1) + self.Delannoy_number(m,n-1)

    
    def brute_force_complexity(self, m, n):
        """Complexity of the BnB algorithm"""
        if self.simplified:
            return self.simple_complexity(m, n)
        return self.Delannoy_number(m, n)

    
    @lru_cache(maxsize=2**10)
    def signature_x(self, I, J):
        """Compute signature of path x on interval [I,J] with memoization"""
        i_0 = I - self.i0
        i_N = J - self.i0
        if self.rough_path:
            #print(self.x_inv[i_0].shape, self.x[i_N].shape, self.width, self.depth)
            return iisignature.sigcombine(self.x_inv[i_0], self.x[i_N], self.width, self.depth)
        return iisignature_sig(self.x[i_0:i_N+1], self.depth)

    @lru_cache(maxsize=2**10)
    def signature_y(self, I, J):
        """Compute signature of path y on interval [I,J] with memoization"""
        j_0 = I - self.j0
        j_N = J - self.j0
        if self.rough_path:
            return iisignature.sigcombine(self.y_inv[j_0], self.y[j_N], self.width, self.depth)
        return iisignature_sig(self.y[j_0:j_N+1], self.depth)

    @lru_cache(maxsize=2**10)
    def signature_norm_diff(self, i, j, I, J):
        """Compute the norm of the signature difference between S(x[i:J]) and S(y[j:J]) with memoization"""
        sig_x = self.signature_x(i, I)
        sig_y = self.signature_y(j, J)
        return pvar_tools.vector_norm(sig_x, sig_y, self.norm)

    def projections_warp2paths(self, warp):
        """Given a warping path in the lattice returns:
           1) index_x_reparam: time parametrization driven by warp of the input path x
           2) index_y_reparam: time parametrization driven by warp of the input path y
           3) projections: "hashed" tuple of tuples of x-y pair of sub-pathlet coordinates for memoization
        """
        index_x_reparam = []
        index_y_reparam = []
        projections = []
        for i,j in warp:
            index_x_reparam.append(i)
            index_y_reparam.append(j)
            projections.append((tuple(self.x[i,:]), tuple(self.y[j,:])))
        return index_x_reparam, index_y_reparam, hash(tuple(projections)) # hashing dic keys for quicker look-up and better memory

    def distance(self, warp, optim_partition=False):
        """computes warped p-variation along one path with dynamic programming algorithm"""

        length = len(warp)
        index_x_reparam, index_y_reparam, projections = self.projections_warp2paths(warp)

        # record memoization size and monitor organization
        passed_time = time.time()-self.initial_time
        self.total_size_of_pvar_cache[passed_time] = len(self.pvar_dist_mem)
        self.pvar_mem_org[projections].append(passed_time)

        if (projections in self.pvar_dist_mem) and (not optim_partition):          
            return self.pvar_dist_mem[projections]

        def dist(a, b):
            i_0, i_N = index_x_reparam[a], index_x_reparam[b]
            j_0, j_N = index_y_reparam[a], index_y_reparam[b]
            return self.signature_norm_diff(i_0+self.i0, 
                                            j_0+self.j0, 
                                            i_N+self.i0, 
                                            j_N+self.j0)
            
        res = pvar_backend.p_var_backbone_ref(length, self.p, dist, optim_partition)
        self.pvar_dist_mem[projections] = res
        return res        
           
    def sense(self):
        """objective: minimize over all paths"""
        return pybnb.minimize

    def objective(self):
        """ The search space is not all paths in the tree, but only complete paths, 
            i.e. paths terminating at (m,n), the very last node for all branches.
            by returning self.distance(self.path) only when self.path is a complete 
            path will ensure to optimise over the right search space (instead of 
            optimising over all possible partial paths on the tree).
        """

        if self.path[-1] == (self.m-1,self.n-1):
            val, _ = self.distance(self.path)
        else:
            val = self.infeasible_objective()

        return val

    def bound1(self, warp):
        """inf_w(d_pvar(x \circ w_x, y \circ w_y)) >= ||S(x \circ w_x) - S(y \circ w_y)||"""

        i, j = warp[0]
        I, J = warp[-1]

        return self.signature_norm_diff(i+self.i0, 
                                        j+self.j0, 
                                        I+self.i0, 
                                        J+self.j0)

    def bound2(self, warp):
        """warped p-variation distance along path so far"""

        b, _ = self.distance(warp)
        return b

    @lru_cache(maxsize=None)
    def bound3_precomputation(self, I, J):

        i = I - self.i0
        j = J - self.j0

        if (i>self.bc) and (j>self.bc): 

            sub_x = self.x[i:]
            sub_x_inv = self.x_inv[i:]
            sub_y = self.y[j:]
            sub_y_inv = self.y_inv[j:]

            sub_problem = BnBWarping(x=sub_x, 
                                     x_inv=sub_x_inv,
                                     y=sub_y, 
                                     y_inv=sub_y_inv,
                                     p=self.p, 
                                     norm=self.norm, 
                                     root_node=(i,j), 
                                     bc=1,
                                     width=self.width,
                                     pvar_dist_mem=self.pvar_dist_mem, 
                                     cache_size=self.cache_size,
                                     rough_path = self.rough_path,

                                     plot_2d=self.plot_2d,
                                     record_path=False, 
                                     pvar_mem_org=self.pvar_mem_org, 
                                     initial_time=self.initial_time,
                                     use_bound1=self.use_bound1, 
                                     use_bound2=self.use_bound2, 
                                     use_bound3=self.use_bound3,
                                     )

            return pybnb.Solver().solve(sub_problem, log=None, queue_strategy='depth').objective

        return 0.

    def bound3(self, warp):
        """Dynamic programming bound (using solution to sub-problems)"""

        i,j = warp[-1] # current position in lattice

        return self.bound3_precomputation(i+self.i0,j+self.j0)

    def compute_bound(self, warp):
        # Cascading better and better bounds (when necessary)

        if not self.use_bound1:
            return self.unbounded_objective()

        b = self.bound1(warp)

        if (b < self.best_node_value) and (self.use_bound2):
            b = self.bound2(warp)
            if (b < self.best_node_value) and (self.use_bound3):
                b = (b**self.p + self.bound3(warp)**self.p)**(1./self.p)

        return b

    def bound(self):
        """ This function is evaluated at a partial path and needs to be a lower bound on any complete 
            path originating from it, so it can decide if the search needs to continue 
            along a partial path based on the best known objective.
        """
        return self.compute_bound(self.path)
        
    def notify_new_best_node(self, node, current):   
        self.best_node_value = node.objective

    def save_state(self, node):
        node.state = list(self.path)

    def load_state(self, node):
        self.path = node.state

    def branch(self):
        
        i,j = self.path[-1]

        if (i==self.m-1) and (j<self.n-1):
            child = pybnb.Node()
            child.state = self.path + [(i,j+1)]
            
            # record edges and nodes
            if self.record_path:
                self.nodes.append(tuple(child.state))
                self.edges.append((tuple(self.path), tuple(child.state)))

            yield child
        
        elif (i<self.m-1) and (j==self.n-1):
            child = pybnb.Node()
            child.state = self.path + [(i+1,j)]

            # record edges and nodes
            if self.record_path:
                self.nodes.append(tuple(child.state))
                self.edges.append((tuple(self.path), tuple(child.state)))

            yield child
        
        elif (i<self.m-1) and (j<self.n-1):
            nodes_update = [(i+1,j+1), (i,j+1), (i+1,j)]
            for v in nodes_update:
                child = pybnb.Node()
                child.state = self.path + [v]

                # record edges and nodes
                if self.record_path:
                    self.nodes.append(tuple(child.state))
                    self.edges.append((tuple(self.path), tuple(child.state)))

                yield child

        #if self.simplified: # x is "longer" than y, so m>n
        #    if (j == i + self.n-self.m) and (i<self.m-1) and (j<self.n-1):
        #        child = pybnb.Node()
        #        child.state = self.path + [(i+1,j+1)]
        #        yield child
        #    elif (i<self.m-1) and (j==self.n-1):
        #        child = pybnb.Node()
        #        child.state = self.path + [(i+1,j)]
        #        yield child
        #    elif (j > i + self.n-self.m) and (i<self.m-1) and (j<self.n-1):
        #        nodes_update = [(i+1,j), (i+1,j+1)]
        #        for v in nodes_update:
        #            child = pybnb.Node()
        #            child.state = self.path + [v]
        #            yield child

    def plot_alignment(self, best_warp):
        """Plot function (for visualization only)"""
        if self.plot_2d:
            plt.plot(self.x.T[0], self.x.T[1], 'bo-' ,label = 'x')
            plt.plot(self.y.T[0], self.y.T[1], 'g^-', label = 'y')
        else:
            plt.plot(self.x, 'bo-' ,label = 'x')
            plt.plot(self.y, 'g^-', label = 'y')
        plt.title('Alignment')
        plt.legend()
        for map_x, map_y in best_warp:
            if self.plot_2d:
                plt.plot([self.x[map_x][0], self.y[map_y][0]], [self.x[map_x][1], self.y[map_y][1]], 'r')
            else:
                plt.plot([map_x, map_y], [self.x[map_x], self.y[map_y]], 'r')