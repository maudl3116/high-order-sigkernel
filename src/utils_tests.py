import abc
import numpy as np
import roughpy as rp
from .utils import PAB, Sig, zero_last_level, get_ctx


def inner_prod_np(t1, t2):
    return np.sum(t1.__array__()*t2.__array__())

class RoughSigKerRP():
    
    """ The RoughSigKerRP object: a class to compute signature kernels and adjoints from piecewise abelian paths with RoughPy """
    
    def __init__(self, S, T, degree=1):
        """ Initialize the RoughSigKer class.
        
        Parameters
        ----------
        
        S        : int, the number of points to join in the partition of the time interval of X
        T        : int, the number of points to join in the partition of the time interval of Y
        degree   : int, the order of the PDE 
        

        """
        self.degree = degree
        self.S = S
        self.T = T
    
    def computeRP(self, X, Y, trunc=12, times=1, coord=0):
        
        """ Approximate the signature kernel and the adjoints using truncated signatures of piecewise abelian paths.
        
        Parameters
        ----------
        X        : array (L_x,D)
        Y        : array (L_y,D)
        trunc    : int, the truncation of signatures,  default=12. 
        times    : int or list, where to return the solution
        coord    : int or list of coordinates to return, default=0 returns only the kernel
        
        Returns
        ----------
        adj_x_y  : array, the adjoint of the signature of X applied to the signature of Y at times and coord.
        adj_y_x  : array, the adjoint of the signature of Y applied to the signature of X at times and coord.
        """
        
        if isinstance(times, int):
            times = [times]
        if isinstance(coord, int):    
            coord = [coord]
        
        S_X, S_Y = Sig(X, step=self.S, m=self.degree, trunc=trunc), Sig(Y, step=self.T, m=self.degree, trunc=trunc)
        
        adj_x_y  = np.array([[rp.adjoint_to_free_multiply(S_X[i], S_Y[j]).__array__() for j in range(len(S_Y))]\
                             for i in range(len(S_X))])
        
        adj_y_x  = np.array([[rp.adjoint_to_free_multiply(S_Y[j], S_X[i]).__array__() for j in range(len(S_Y))]\
                             for i in range(len(S_X))])
        
      
        if len(times)==1 and len(coord)==1:
            return adj_x_y[times[0], times[0], coord[0]], adj_y_x[times[0], times[0], coord[0]]
        else:
            return  np.squeeze(adj_x_y[np.ix_(times, times, coord)]),  np.squeeze(adj_y_x[np.ix_(times, times, coord)])
    
    def kerRP(self, X, Y, trunc=12, times=1):
        
        """ Approximate the signature kernel using truncated signatures of piecewise abelian paths.
        
        Parameters
        ----------
        X        : array (L_x,D)
        Y        : array (L_y,D)
        trunc    : int, the truncation of signatures,  default=12. 
        times    : where to return the solution
        
        Returns
        ----------
        K        : array, the signature kernel of X and Y at times.
        """
       
        S_X, S_Y = Sig(X, step=self.S, m=self.degree, trunc=trunc), Sig(Y, step=self.T, m=self.degree, trunc=trunc)
        
        K = np.array([[inner_prod_np(S_X[i],S_Y[j]) for j in range(len(S_Y))]\
                             for i in range(len(S_X))])

        if isinstance(times, int):
            return K[times, times]
        elif isinstance(times, list) and len(times)==1:
            return K[times[0], times[0]]
        else:
            return K[np.ix_(times, times)]
        
   
