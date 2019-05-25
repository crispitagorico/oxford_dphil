import timeit
setup = """\
import numpy as np
#from tjl_dense_numpy_tensor import *
#from tjl_hall_numpy_lie import *
#
from tjl_dense_numpy_tensor import tensor_log, tensor_multiply, tensor_exp, stream2sigtensor, brownian
from tjl_hall_numpy_lie import hall_basis, l2t, t2l
import tjl_hall_numpy_lie
import tjl_dense_numpy_tensor
import tosig as ts
from esig import tosig as ets
import tjl_timer

def cbh (l1, l2, depth, width):
    t2l(
        tensor_log(
            tensor_multiply(
                tensor_exp(
                    l2t(
                        l1,width,depth
                        ),depth
                    ),
                tensor_exp(
                    l2t(
                        l1,width,depth
                        ),depth
                    ),depth
                ),depth
            )
        )

width = 2
depth = 10
# intialise the caches
hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(width, depth)
l1 = t2l(tensor_log(stream2sigtensor(brownian(20, width), depth), depth))
l2t(l1, width, depth)
"""

command_hall_set = 'hall_set, degrees, degree_boundaries, reverse_map, width = h.;all_basis(width, depth)'
command_l2t='l2t(l1, width, depth)'
command_cbh='cbh(l1, l1, width, depth)'

timing = timeit.timeit(command_cbh , setup, number = 1)
print(timing)



##print([hb_to_string(x, width, depth) for x, y in enumerate(hall_set)])
##print(lie_to_string(t2l(l2t(prod(4, 32, width, depth), width, depth)), width, depth))

#print("\n lie elements being generated; max blob size =", blob_size(width,depth), "\n")
#l1 = t2l(tensor_log(stream2sigtensor(brownian(20, width), depth), depth))
#l2 = t2l(tensor_log(stream2sigtensor(brownian(20, width), depth), depth))

#print("\n lie elements generated; max blob size =", blob_size(width,depth), "\n")

#t2l(
#    tensor_log(
#        tensor_multiply(
#            tensor_exp(
#                l2t(
#                    l1,width,depth
#                    ),depth
#                ),
#            tensor_exp(
#                l2t(
#                    l1,width,depth
#                    ),depth
#                ),depth
#            ),depth
#        )
#    )

#print("\ncbh lie element generated; max blob size =", hall_set.size, "\n")

#t2l(
#    tensor_log(
#        tensor_multiply(
#            tensor_exp(
#                l2t(
#                    l1,width,depth
#                    ),depth
#                ),
#            tensor_exp(
#                l2t(
#                    l1,width,depth
#                    ),depth
#                ),depth
#            ),depth
#        )
#    )

#print("\ncbh lie element generated; max blob size =", hall_set.size, "\n")

#"""
##print(tensor_sub(l2t(t2l(t), width, depth), t))
##print(np.sum(tensor_sub(l2t(t2l(t), width, depth), t)[2:] ** 2) < 1e-30)

#print(ts.sigkeys(width, depth))
#stream = brownian(100, width)
#print(ts.stream2sig(stream,depth))
#print(np.max(np.abs(ets.stream2sig(stream,depth)-ts.stream2sig(stream,depth))))
#print(ts.stream2logsig(stream,depth))
#print(np.max(np.abs(ets.stream2logsig(stream,depth)-ts.stream2logsig(stream,depth))))
#"""