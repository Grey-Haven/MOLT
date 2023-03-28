import numpy as np
import numba as nb


@nb.njit([nb.void(nb.float64[:])], cache= False, boundscheck = False)
def periodic_extension(v):
    """
    Fills the ghost region of an array "v" using periodic copies.
    
    Assumes that v is a 1-D array which has 2 ghost points on each end.
    
    Note: v includes the extension, so indices are adjusted accordingly.
    """
    # Left region
    v[0:2] = v[-5:-3]

    # Right region
    v[-2:] = v[3:5]
    
    return None


@nb.njit([nb.void(nb.float64[:])], cache= False, boundscheck = False)
def polynomial_extension(v):
    """
    Fills the ghost region of an array "v" using polynomial extrapolation.

    Assumes that v is a 1-D array which has 2 ghost points on each end.
    
    Note: v includes the extension, so indices are adjusted accordingly.
    """
    # Left region
    v[0] = 15*v[2]- 40*v[3] + 45*v[4] -24*v[5] + 5*v[6]
    v[1] =  5*v[2]- 10*v[3] + 10*v[4] - 5*v[5] +   v[6]

    # Right region
    v[-2] =  5*v[-3] - 10*v[-4] + 10*v[-5] -  5*v[-6] +   v[-7]
    v[-1] = 15*v[-3] - 40*v[-4] + 45*v[-5] - 24*v[-6] + 5*v[-7]
    
    return None