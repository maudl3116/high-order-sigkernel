import abc
import numpy as np
import roughpy as rp
from .utils import PAB, Sig, zero_last_level, get_ctx
from .integrate import _goursat_integrate, _characteristics_integrate

#================================================================================================================================
# Class that provides the skeleton of a SigKer PDE solver
#================================================================================================================================
class Solver(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def _init_state(self, X, Y):
        pass
    
    def integrate(self, X, Y):
        pass
        
#================================================================================================================================
# SigKer PDE solvers
#================================================================================================================================        
class CharacteristicsSolver(Solver):
    
    def _init_state(self, X, Y, S_X, S_Y):
        
        M, N, D_lie = len(X)+1, len(Y)+1, X[0].__array__().shape[0]  
        ctx = get_ctx(X[0])
        
        # to save intermediate states 
        zero_tensor = rp.FreeTensor(np.zeros(D_lie), ctx=ctx)
        adj_x_y = [[zero_tensor]*N]*M     
        adj_y_x = [[zero_tensor]*N]*M
     
        one = np.zeros(D_lie)
        one[0] = 1.

        adj_x_y[0][0] = rp.FreeTensor(one, ctx=ctx)
        for j in range(1,N):
            adj_x_y[0][j] = S_Y[j-1]
            adj_y_x[0][j] = rp.FreeTensor(one, ctx=ctx)

        adj_y_x[0][0] = rp.FreeTensor(one, ctx=ctx)
        for i in range(1,M):
            adj_y_x[i][0] = S_X[i-1]
            adj_x_y[i][0] = rp.FreeTensor(one, ctx=ctx)
            
        return adj_x_y, adj_y_x
    
    def integrate(self, X, Y, S_X, S_Y, refine=1):
        
        # repeat the tensors so that the PDE solver can do refine intermediate steps
        X = [x*(1./refine) for x in X for k in range(refine)]
        Y = [y*(1./refine) for y in Y for k in range(refine)]
        S_X = [x*(1./refine) for x in S_X for k in range(refine)]
        S_Y = [y*(1./refine) for y in S_Y for k in range(refine)]    
        
        # initialise the states
        adj_x_y, adj_y_x = self._init_state(X, Y, S_X, S_Y)

        # integrate
        adj_x_y, adj_y_x = _characteristics_integrate(X, Y, adj_x_y, adj_y_x)

        M, N = len(X)+1, len(Y)+1
        
        # convert roughpy objects into numpy array
        adj_x_y = np.array([[adj_x_y[i][j].__array__() for j in range(N)] for i in range(M)]) 
        
        adj_y_x = np.array([[adj_y_x[i][j].__array__() for j in range(N)] for i in range(M)])

        return adj_x_y, adj_y_x

class GoursatSolver(Solver):
    
    def _init_state(self, X, Y, S_X, S_Y):
        
        M, N, D_lie = len(X)+1, len(Y)+1, X[0].__array__().shape[0]  
        ctx = get_ctx(X[0])
        
        # to save intermediate states 
        K = np.zeros((M, N), dtype=np.float64) 

        zero_tensor = rp.FreeTensor(np.zeros(D_lie), ctx=ctx)
        adj_x_y = [[zero_tensor]*N]*M     
        adj_y_x = [[zero_tensor]*N]*M

        # initialize states
        K[0,:] = 1.
        K[:,0] = 1.

        Y_sub = zero_last_level(S_Y, ctx, remove_one=True)
        for j in range(1,N):
            adj_x_y[0][j] = Y_sub[j-1]
            
        X_sub = zero_last_level(S_X, ctx, remove_one=True)
        for i in range(1,M):
            adj_y_x[i][0] = X_sub[i-1]
            
        return K, adj_x_y, adj_y_x
        
    def integrate(self, X, Y, S_X, S_Y, refine=1):
 
        # repeat the tensors so that the PDE solver can do refine intermediate steps
        X = [x*(1./refine) for x in X for k in range(refine)]
        Y = [y*(1./refine) for y in Y for k in range(refine)]
        S_X = [x*(1./refine) for x in S_X for k in range(refine)]
        S_Y = [y*(1./refine) for y in S_Y for k in range(refine)]

        # initialise the states
        K, adj_x_y, adj_y_x = self._init_state(X, Y, S_X, S_Y)
        
        # integrate
        K, adj_x_y, adj_y_x = _goursat_integrate(X, Y, K, adj_x_y, adj_y_x)
        
        D, degree = X[0].width, X[0].max_degree
        
        M, N = len(X)+1, len(Y)+1
        
        # convert roughpy objects into numpy array
        
        adj_x_y = np.array([[adj_x_y[i][j].__array__()[:-D**degree] for j in range(N)] for i in range(M)]) 
        
        adj_y_x = np.array([[adj_y_x[i][j].__array__()[:-D**degree] for j in range(N)] for i in range(M)])
        
        adj_x_y[...,0] = K
        
        adj_y_x[...,0] = K

        return adj_x_y, adj_y_x
    
SOLVERS = {
    'characteristics': CharacteristicsSolver,
    'goursat': GoursatSolver
}    

#================================================================================================================================
# Main class to compute the signature kernel of two paths. 
#================================================================================================================================

class RoughSigKer():
    
    """ The RoughSigKer object: a class to compute signature kernels from piecewise abelian paths """
    
    def __init__(self, S, T, degree=1, solver='goursat'):
        """ Initialize the RoughSigKer class.
        
        Parameters
        ----------
        
        S        : int, the number of intervals to join in the partition of the time interval of X
        T        : int, the number of intervals to join in the partition of the time interval of Y
        degree   : int, the order of the PDE 
        solver   : {'goursat', 'characteristics'},  default='goursat'. 

        """
        self.degree = degree
        self.S = S
        self.T = T
        self.solver = SOLVERS[solver]()
    
    def solvePDE(self, X, Y, times=1, coord=0):
        
        """ Approximate the signature kernel and the adjoints using a log-PDE method.
        
        Parameters
        ----------
        X        : array (L_x,D)
        Y        : array (L_y,D)
        times    : int or list of ints, where to return the solution
        coord    : int or list of ints, coordinates to return, default=0 returns only the kernel
        
        Returns
        ----------
        adj_x_y  : array, the adjoint of the signature of X applied to the signature of Y at times and coord.
        adj_y_x  : array, the adjoint of the signature of Y applied to the signature of X at times and coord.
        """
        
        if isinstance(times, int):
            times = [times]
        if isinstance(coord, int):    
            coord = [coord]
        
        X_PAB, Y_PAB = PAB(X, step=self.S, m=self.degree), PAB(Y, step=self.T, m=self.degree)
        
        S_X, S_Y = Sig(X, step=self.S, m=self.degree), Sig(Y, step=self.T, m=self.degree)
        
        adj_x_y, adj_y_x = self.solver.integrate(X_PAB, Y_PAB, S_X, S_Y)
        
        if len(times)==1 and len(coord)==1:
            return adj_x_y[times[0], times[0], coord[0]], adj_y_x[times[0], times[0], coord[0]]
        else:
            return  np.squeeze(adj_x_y[np.ix_(times, times, coord)]),  np.squeeze(adj_y_x[np.ix_(times, times, coord)])
    
    def kerPDE(self, X, Y, times=1, refine=1):
        
        """ Approximate the signature kernel using a log-PDE method.
        
        Parameters
        ----------
        X        : array (L_x,D)
        Y        : array (L_y,D)
        times    : int or list, where to return the solution
        refine   : int, the number of intermediate steps from the numerical solver
        
        Returns
        ----------
        K        : array, the signature kernel of X and Y at times.
        """
        
        X_PAB, Y_PAB = PAB(X, step=self.S, m=self.degree), PAB(Y, step=self.T, m=self.degree)
        
        S_X, S_Y = Sig(X, step=self.S, m=self.degree), Sig(Y, step=self.T, m=self.degree)
        
        self.X_PAB, self.Y_PAB = X_PAB, Y_PAB
        
        adj_x_y, adj_y_x = self.solver.integrate(X_PAB, Y_PAB, S_X, S_Y, refine=refine)

        if isinstance(times, int):
            K = adj_x_y[times, times, 0]
        elif isinstance(times, list) and len(times)==1:
            K = adj_x_y[times[0], times[0], 0]
        else:
            K = adj_x_y[np.ix_(times, times)][:,:,0]
        return K
   
