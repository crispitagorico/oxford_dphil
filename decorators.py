import functools
import copy


def lru_cache(maxsize=None):
    """Decorator that handles memoization.
    
    Parameters
    ----------
    maxsize : int or None, optional
        Ignored. For compatibility with functools.
    
    """
        
    class Memoization:
        def __init__(self, f):
            self.f = f
            self.cache = {}
        def __call__(self, *args):

            if not args in self.cache:
                self.cache[args] = self.f(*args)
            
            return copy.deepcopy(self.cache[args])

    return Memoization

def accepts(*types, **kwtypes):
    """Decorator to restrict the argument types of a function."""
    
    def check_accepts(f):
        @functools.wraps(f)
        def new_f(*args, **kwds):
            for key in kwds:
                if key not in kwtypes:
                    continue
                    
                if kwtypes[key] is None:
                    continue
                

                assert isinstance(kwds[key], kwtypes[key]), \
                        "arg %r does not match %s" % (kwds[key], kwtypes[key])
                 
            for (a, t) in zip(args, types):
                if t is None:
                    continue
                
                assert isinstance(a, t), \
                       "arg %r does not match %s" % (a,t)
           
            return f(*args, **kwds)

        return new_f

    return check_accepts