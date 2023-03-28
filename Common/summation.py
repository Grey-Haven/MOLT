import numpy as np
import numba as nb

@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                 nb.float64, nb.float64)], 
                 cache = False, boundscheck = False)
def fast_convolution(I_L, I_R, alpha, dx):

    N = I_L.size
    
    # Precompute the recursion weight
    weight = np.exp( -alpha*dx )
    
    # Perform the sweeps to the right
    for i in range(1,N):
        
        I_L[i] = weight*I_L[i-1] + I_L[i]
    
    # Perform the sweeps to the left
    for i in range(N-2,-1,-1):
        
        I_R[i] = weight*I_R[i+1] + I_R[i]
    
    return None