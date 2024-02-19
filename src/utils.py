import numpy as np
import roughpy as rp

def get_ctx(X):
    return rp.get_context(X.width, X.max_degree, X.dtype)

def Int(step,i):
    return rp.RealInterval(step*i, step*(i+1))

def PAB(X, step, m):
    """ Construct the piecewise abelian approximation of X of degree m on a partition D={t_i}
        where t_i+1 - t_i = step
        
        Parameters
        ----------
        X        : array (L_x,D), the input path
        size     : int, the size of the partition
        m        : int, degree of the log signatures
        
        Output
        ----------
        X_PAB    : piecewise abelian approximation of X of degree m on a partition D={t_i}
        where t_i+1 - t_i = step        
    """
    X_inc = np.diff(X, axis=0)
    
    L = X_inc.shape[0]
    D = X_inc.shape[1]
    
    I = rp.RealInterval(0., L)
    
    s = rp.LieIncrementStream.from_increments(X_inc, width=D, depth=m)

#     X_PAB = rp.PiecewiseAbelianStream.construct( [ (Int(step,i), s.log_signature(Int(step,i), depth=m) ) \
#                                                   for i in range(L//step)\
#                                                   ] , width=D, depth=m, dtype='DPReal'\
#                                                )

    X_PAB = [ s.log_signature(Int(step,i), depth=m) for i in range(L//step) ]

    ctx = rp.get_context(D, m, X_PAB[0].dtype)
    
    X_PAB = [ ctx.lie_to_tensor(x) for x in X_PAB ]

    return X_PAB

def Sig(X, step, m, trunc=None):
    """ Compute the pathwise signature of X of degree m on a partition D={t_i}
        where t_i+1 - t_i = step
        
        Parameters
        ----------
        X        : array (L_x,D), the input path
        size     : int, the size of the partition
        m        : int, degree of the log signatures
        trunc    : 
        
        Output
        ----------
        Sig_X    : pathwise signature of X of degree trunc on a partition D={t_i}
        where t_i+1 - t_i = step
        
    """
    
    if trunc is None:
        trunc = m
    
    X_inc = np.diff(X, axis=0)
    
    L = X_inc.shape[0]
    D = X_inc.shape[1]
    
    I = rp.RealInterval(0., L)
    
    s = rp.LieIncrementStream.from_increments(X_inc, width=D, depth=m)

    Sig_X = [ s.signature(rp.RealInterval(0, step*(i+1)), depth=trunc) for i in range(L//step) ]


    return Sig_X


def zero_last_level(X, ctx, remove_one=False):
    
    D, degree = X[0].width, X[0].max_degree
    
    X_sub = []
    
    for x in X:
        x = x.__array__()
        x[-D**degree:] = 0.
        if remove_one:
            x[0]=0.
        x = rp.FreeTensor(x, ctx=ctx)
        X_sub.append(x)

    return X_sub

