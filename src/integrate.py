import numpy as np
import roughpy as rp
from .utils import zero_last_level, get_ctx

#================================================================================================================================
# Utilities
#================================================================================================================================       
def right_adj(A,C, tensor_context):

    ant_A = A.antipode()
    ant_C = C.antipode()
    
    left_adj = rp.adjoint_to_free_multiply(ant_A, ant_C)

    left_adj = rp.FreeTensor(left_adj.__array__(), ctx=tensor_context)  
    
    adj_A_C = left_adj.antipode()
    
    return adj_A_C
    
def inner_prod(X,Y):
    return np.sum(X.__array__()*Y.__array__())
    
def eval_adj(adj_x_y, adj_y_x, x, y, tensor_context):
    r_x_y = right_adj(x, y, tensor_context)  # change context
    r_y_x = right_adj(y, x, tensor_context)  # change context

    return inner_prod(adj_y_x, r_x_y) + inner_prod(adj_x_y, r_y_x)

def tensor(scalar, D, context):
    t = np.zeros(D)
    t[0] = scalar
    return rp.FreeTensor(t, ctx=context)

def tensor_mul(scalar, D, context):
    t = np.zeros(D)
    t[0] = scalar
    return rp.FreeTensor(t, ctx=context)

def first_coord(t):
    return t.__array__()[0]
           
#================================================================================================================================
# Goursat integrator
#================================================================================================================================          
def _goursat_integrate(X, Y, K, adj_x_y, adj_y_x):

    M = len(X)
    N = len(Y)
    D = X[0].__array__().shape[0]
    ctx = get_ctx(X[0])
    
    # set to zero the last degree (ideally should be a functionality of roughpy)
    XX, YY = zero_last_level(X, ctx), zero_last_level(Y, ctx)
        
    for i in range(M):
        for j in range(N):
            
     
            #phi
            adj_y_x[i+1][j+1] =  adj_y_x[i][j+1] + XX[i].__mul__(K[i,j])\
                                + adj_y_x[i][j+1].__mul__(XX[i])\
                                + rp.adjoint_to_free_multiply(adj_x_y[i][j+1], XX[i])\
                                - tensor(inner_prod(adj_x_y[i][j+1], XX[i]), D, ctx)

            #psis
            adj_x_y[i+1][j+1] =  adj_x_y[i+1][j] + YY[j].__mul__(K[i,j])\
                                + adj_x_y[i+1][j].__mul__(YY[j])\
                                + rp.adjoint_to_free_multiply(adj_y_x[i+1][j], YY[j])\
                                - tensor(inner_prod(adj_y_x[i+1][j], YY[j]), D, ctx)

            eval_adj_ = eval_adj(adj_x_y[i][j], adj_y_x[i][j], X[i], Y[j], ctx)
            

            # the kernel equation
            next_eval_adj = eval_adj(adj_x_y[i+1][j+1], adj_y_x[i+1][j+1], X[i], Y[j], ctx)
            temp_2 =  eval_adj(adj_x_y[i][j+1], adj_y_x[i][j+1], X[i], Y[j], ctx)
            temp_3 = eval_adj(adj_x_y[i+1][j], adj_y_x[i+1][j], X[i], Y[j], ctx)
   
            G = inner_prod(X[i],Y[j])
            f_1 = K[i,j] * G + eval_adj_
            f_2 = K[i,j+1] * G + temp_2 
            f_3 = K[i+1,j] * G + temp_3 

            u_p = K[i+1,j] + K[i,j+1] - K[i,j] + f_1
            f_p = u_p * G + next_eval_adj


            K[i+1,j+1] =  K[i+1,j] + K[i,j+1] - K[i,j] + (1./4)*(f_1 + f_2 + f_3 + f_p) 

    return K, adj_x_y, adj_y_x

#================================================================================================================================
# Characteristics integrator
#================================================================================================================================      
def _characteristics_integrate(X, Y, adj_x_y, adj_y_x):
  
    M = len(X)
    N = len(Y)

    for i in range(M):
        for j in range(N):
            
            #phi
            adj_y_x[i+1][j+1] =  adj_y_x[i][j+1] - X[i].__mul__(first_coord(adj_y_x[i][j]))\
                                + adj_y_x[i][j+1].__mul__(X[i])\
                                + rp.adjoint_to_free_multiply(adj_x_y[i][j+1], X[i])
 
            #psi
            adj_x_y[i+1][j+1] =  adj_x_y[i+1][j] - Y[j].__mul__(first_coord(adj_x_y[i][j]))\
                                + adj_x_y[i+1][j].__mul__(Y[j])\
                                + rp.adjoint_to_free_multiply(adj_y_x[i+1][j], Y[j])


    return adj_x_y, adj_y_x
