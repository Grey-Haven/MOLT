import numpy as np
import numba as nb
from operators import *

#
# Backwards Difference Methods
# 


#
#
# Periodic functions
#
#

#-------------------------------------------------------------
# Advance for the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_advance_per(v, src_data, x, y, t, 
                     dx, dy, dt, c, beta):
    """
    Calculates a solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_per(v[3,:,:], tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized advance function for the 2D heating test
#
@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF1_advance_per(v, src_data, x, y, t, 
                     dx, dy, dt, c, beta):
    """
    Calculates a solution to the wave equation using the first-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            # There are three time levels here
            R[i,j] = 2*v[1,i,j] - v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_per(v[2,:,:], tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized advance for the parabolic equation
#
@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_advance_per_parabolic(v, src_data, x, y, t, 
                               dx, dy, dt, alpha):
    """
    Calculates a solution to the diffusion equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]
           
    # Invert the y operator and apply to R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_x_inverse_per(v[2,:,:], tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None


#-------------------------------------------------------------
# Advance for the x-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddx_advance_per(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_per(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized advance for the parabolic case
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddx_advance_per_parabolic(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, alpha):
    """
    Calculates the ddx of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_ddx_L_x_inverse_per(ddx, tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF3_ddx_advance_per(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the third-order BDF method.
    By third-order, we mean that a third-order difference is used for the u_tt term. The splitting
    error is not addressed by this method. This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = (12/35)*( (26/3)*v[3,i,j] - (19/2)*v[2,i,j] + (14/3)*v[1,i,j] - (11/12)*v[0,i,j] )
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_per(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF4_ddx_advance_per(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the fourth-order BDF method.
    By fourth-order, we mean that a fourth-order difference is used for the u_tt term. The splitting
    error is not addressed by this method. This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/15)*( (77/6)*v[4,i,j] - (107/6)*v[3,i,j] + 13*v[2,i,j] - (61/12)*v[1,i,j] + (5/6)*v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_per(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized advance function for the 2D heating test
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF1_ddx_advance_per(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the first-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            # There are three time levels here
            R[i,j] = 2*v[1,i,j] - v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_per(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#-------------------------------------------------------------
# Advance for the second x-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dxx_advance_per(dxx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the dxx of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_dxx_L_x_inverse_per(dxx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized method for the parabolic case
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dxx_advance_per_parabolic(dxx, v, src_data, x, y, t, 
                         dx, dy, dt, alpha):
    """
    Calculates the dxx of the solution to the diffusion equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_y_inverse_per(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_dxx_L_x_inverse_per(dxx, tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None



#-------------------------------------------------------------
# Advance for the y-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddy_advance_per(ddy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_ddy_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_L_x_inverse_per(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized method for the parabolic case
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddy_advance_per_parabolic(ddy, v, src_data, x, y, t, 
                                   dx, dy, dt, alpha):
    """
    Calculates the ddy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_ddy_L_y_inverse_per(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_x_inverse_per(ddy, tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF3_ddy_advance_per(ddy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the third-order BDF method.
    By third-order, we mean that a third-order difference is used for the u_tt term. The splitting
    error is not addressed by this method. This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (12/35)*( (26/3)*v[3,i,j] - (19/2)*v[2,i,j] + (14/3)*v[1,i,j] - (11/12)*v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_ddy_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_L_x_inverse_per(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF4_ddy_advance_per(ddy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the fourth-order BDF method.
    By fourth-order, we mean that a fourth-order difference is used for the u_tt term. The splitting
    error is not addressed by this method. This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/15)*( (77/6)*v[4,i,j] - (107/6)*v[3,i,j] + 13*v[2,i,j] - (61/12)*v[1,i,j] + (5/6)*v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_ddy_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_L_x_inverse_per(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized method for the 2D heating test
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF1_ddy_advance_per(ddy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the first-order BDF method. 
    This function accepts the mesh data v and src_fcn.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            # There are three time levels here
            R[i,j] = 2*v[1,i,j] - v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_ddy_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_L_x_inverse_per(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#-------------------------------------------------------------
# Advance for the second y-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dyy_advance_per(dyy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the dyy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
#     # Invert the y operator and apply dyy to R, storing in tmp
#     get_dyy_L_y_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
#     # Invert the x operator on tmp, then store in the derivative array
#     get_L_x_inverse_per(dyy, tmp, x, y, dx, dy, dt, c, beta)

    # Invert the x operator on R, storing in tmp
    get_L_x_inverse_per(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the y operator on tmp, apply dyy then store in the derivative array
    get_dyy_L_y_inverse_per(dyy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized method for the parabolic case
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dyy_advance_per_parabolic(dyy, v, src_data, x, y, t, 
                                   dx, dy, dt, alpha):
    """
    Calculates the dyy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]

    # Invert the x operator on R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_x_inverse_per(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the y operator on tmp, apply dyy then store in the derivative array
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_dyy_L_y_inverse_per(dyy, tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None


#
#
# Dirichlet functions (Homogeneous case)
#
#

#-------------------------------------------------------------
# Advance for the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_advance_dir(v, src_data, x, y, t, 
                     dx, dy, dt, c, beta):
    """
    Calculates a solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_dir(v[3,:,:], tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None


#
# Specialized advance function for the expanding beam problem
#
@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF1_advance_dir(v, src_data, x, y, t, 
                     dx, dy, dt, c, beta):
    """
    Calculates a solution to the wave equation using the first-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            # There are three time levels here
            R[i,j] = 2*v[1,i,j] - v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_dir(v[2,:,:], tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized advance for the parabolic equation
#
@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_advance_dir_parabolic(v, src_data, x, y, t, 
                               dx, dy, dt, alpha):
    """
    Calculates a solution to the diffusion equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_x_inverse_dir(v[2,:,:], tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None




#-------------------------------------------------------------
# Advance for the x-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddx_advance_dir(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the second order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_dir(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None



#
# Specialized advance function for the expanding beam problem
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF1_ddx_advance_dir(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the first-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            # There are three time levels here
            R[i,j] = 2*v[1,i,j] - v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_dir(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None


#
# Specialized advance for the parabolic case
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddx_advance_dir_parabolic(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, alpha):
    """
    Calculates the ddx of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_ddx_L_x_inverse_dir(ddx, tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF3_ddx_advance_dir(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the third-order BDF method.
    By third-order, we mean that a third-order difference is used for the u_tt term. The splitting
    error is not addressed by this method. This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (12/35)*( (26/3)*v[3,i,j] - (19/2)*v[2,i,j] + (14/3)*v[1,i,j] - (11/12)*v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_dir(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF4_ddx_advance_dir(ddx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the fourth-order BDF method.
    By fourth-order, we mean that a fourth-order difference is used for the u_tt term. The splitting
    error is not addressed by this method. This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/15)*( (77/6)*v[4,i,j] - (107/6)*v[3,i,j] + 13*v[2,i,j] - (61/12)*v[1,i,j] + (5/6)*v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_dir(ddx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#-------------------------------------------------------------
# Advance for the second x-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dxx_advance_dir(dxx, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the dxx of the solution to the wave equation using the second order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply dxx to tmp, then store in the derivative array
    get_dxx_L_x_inverse_dir(dxx, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized method for the parabolic case
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dxx_advance_dir_parabolic(dxx, v, src_data, x, y, t, 
                         dx, dy, dt, alpha):
    """
    Calculates the dxx of the solution to the diffusion equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]
    
            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_y_inverse_dir(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_dxx_L_x_inverse_dir(dxx, tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None


#-------------------------------------------------------------
# Advance for the y-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddy_advance_dir(ddy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_fcn.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
#     # Invert the y operator and apply to R, storing in tmp
#     get_ddy_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
#     # Invert the x operator and apply to tmp, then store in the derivative array
#     get_L_x_inverse_dir(ddy, tmp, x, y, dx, dy, dt, c, beta)

    # Change the order of operations
    
    # Invert the x operator and apply to R, then store in tmp the derivative array
    get_L_x_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the y operator and apply to tmp, take ddy, and store in the derivative array
    get_ddy_L_y_inverse_dir(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None


#
# Specialized method for the expanding beam problem
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF1_ddy_advance_dir(ddy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the first-order BDF method. 
    This function accepts the mesh data v and src_fcn.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            # There are three time levels here
            R[i,j] = 2*v[1,i,j] - v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_ddy_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_L_x_inverse_dir(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF3_ddy_advance_dir(ddy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the third-order BDF method.
    By third-order, we mean that a third-order difference is used for the u_tt term. The splitting
    error is not addressed by this method. This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (12/35)*( (26/3)*v[3,i,j] - (19/2)*v[2,i,j] + (14/3)*v[1,i,j] - (11/12)*v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_ddy_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_L_x_inverse_dir(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF4_ddy_advance_dir(ddy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the fourth-order BDF method.
    By fourth-order, we mean that a fourth-order difference is used for the u_tt term. The splitting
    error is not addressed by this method. This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/15)*( (77/6)*v[4,i,j] - (107/6)*v[3,i,j] + 13*v[2,i,j] - (61/12)*v[1,i,j] + (5/6)*v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_ddy_L_y_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_L_x_inverse_dir(ddy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized method for the parabolic case
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddy_advance_dir_parabolic(ddy, v, src_data, x, y, t, 
                                   dx, dy, dt, alpha):
    """
    Calculates the ddy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_ddy_L_y_inverse_dir(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_x_inverse_dir(ddy, tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None

#-------------------------------------------------------------
# Advance for the second y-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dyy_advance_dir(dyy, v, src_data, x, y, t, 
                         dx, dy, dt, c, beta):
    """
    Calculates the dyy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_fcn.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    alpha = beta/(c*dt)
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )

            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
    
    # Invert the x operator and apply to R, then store in tmp the derivative array
    get_L_x_inverse_dir(tmp, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the y operator and apply to tmp, take dyy, and store in the derivative array
    get_dyy_L_y_inverse_dir(dyy, tmp, x, y, dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#
# Specialized method for the parabolic case
#
@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dyy_advance_dir_parabolic(dyy, v, src_data, x, y, t, 
                                   dx, dy, dt, alpha):
    """
    Calculates the dyy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Time history (v doesn't include the extension region)
            R[i,j] = (4/3)*v[1,i,j] - (1/3)*v[0,i,j]

            # Contribution from the source term (at t_{n+1})
            R[i,j] += (2/3)*dt*src_data[i,j]

    # Invert the x operator on R, storing in tmp
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_L_x_inverse_dir(tmp, R, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Invert the y operator on tmp, apply dyy then store in the derivative array
    #
    # Note: The inverse operator methods (used almost everywhere) were
    # designed with wave solvers in mind. Since the def'n of alpha is different
    # for the parabolic equations, we will use the wave solver methods with
    # beta = alpha and c = dt = 1. When we refactor these methods, we will
    # make the inverse functions only a function of alpha for reusability
    get_dyy_L_y_inverse_dir(dyy, tmp, x, y, dx, dy, 1.0, 1.0, alpha)
    
    # Shuffle is performed outside this function
    
    return None




#
#
# Outflow functions
#
#

#-------------------------------------------------------------
# Advance for the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64[:], nb.float64[:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_advance_out(v, src_data, x, y, t,
                     A_nm1_x, B_nm1_x, A_nm1_y, B_nm1_y,
                     bdry_hist_ax, bdry_hist_bx, 
                     bdry_hist_ay, bdry_hist_by,
                     dx, dy, dt, c, beta):
    """
    Calculates a solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size    
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse

    # Time history (v doesn't include the extension region)
    alpha = beta/(c*dt)
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )
            
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]

    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_out(tmp, R, x, y, 
                        A_nm1_y, B_nm1_y,
                        bdry_hist_ay, bdry_hist_by,
                        dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_out(v[3,:,:], tmp, x, y, 
                        A_nm1_x, B_nm1_x,
                        bdry_hist_ax, bdry_hist_bx,
                        dx, dy, dt, c, beta)

    # Shuffle is performed outside this function
    
    return None


#-------------------------------------------------------------
# Advance for the x-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64[:], nb.float64[:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddx_advance_out(ddx, v, src_data, x, y, t,
                         A_nm1_x, B_nm1_x, A_nm1_y, B_nm1_y,
                         bdry_hist_ax, bdry_hist_bx, 
                         bdry_hist_ay, bdry_hist_by,
                         dx, dy, dt, c, beta):
    """
    Calculates the ddx of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size    
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    # Time history (v doesn't include the extension region)
    alpha = beta/(c*dt)
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )
            
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
    
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_out(tmp, R, x, y, 
                        A_nm1_y, B_nm1_y,
                        bdry_hist_ay, bdry_hist_by,
                        dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the derivative array
    get_ddx_L_x_inverse_out(ddx, tmp, x, y, 
                            A_nm1_x, B_nm1_x,
                            bdry_hist_ax, bdry_hist_bx,
                            dx, dy, dt, c, beta)

    # Shuffle is performed outside this function
    
    return None

#-------------------------------------------------------------
# Advance for the second x-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64[:], nb.float64[:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dxx_advance_out(dxx, v, src_data, x, y, t,
                         A_nm1_x, B_nm1_x, A_nm1_y, B_nm1_y,
                         bdry_hist_ax, bdry_hist_bx, 
                         bdry_hist_ay, bdry_hist_by,
                         dx, dy, dt, c, beta):
    """
    Calculates the dxx of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Source function is implicit in the BDF method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size    
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    # Time history (v doesn't include the extension region)
    alpha = beta/(c*dt)
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )
            
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]
    
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_out(tmp, R, x, y, 
                        A_nm1_y, B_nm1_y,
                        bdry_hist_ay, bdry_hist_by,
                        dx, dy, dt, c, beta)
    
    # Invert the x operator and apply dxx to tmp, then store in the derivative array
    get_dxx_L_x_inverse_out(dxx, tmp, x, y, 
                            A_nm1_x, B_nm1_x,
                            bdry_hist_ax, bdry_hist_bx,
                            dx, dy, dt, c, beta)

    # Shuffle is performed outside this function
    
    return None


#-------------------------------------------------------------
# Advance for the y-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64[:], nb.float64[:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_ddy_advance_out(ddy, v, src_data, x, y, t, 
                         A_nm1_x, B_nm1_x, A_nm1_y, B_nm1_y,
                         bdry_hist_ax, bdry_hist_bx, 
                         bdry_hist_ay, bdry_hist_by,
                         dx, dy, dt, c, beta):
    """
    Calculates the ddy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size    
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    # Time history (v doesn't include the extension region)
    alpha = beta/(c*dt)
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )
            
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]

    # Invert the x operator and apply to R, storing in tmp
    get_L_x_inverse_out(tmp, R, x, y, 
                        A_nm1_x, B_nm1_x,
                        bdry_hist_ax, bdry_hist_bx,
                        dx, dy, dt, c, beta)
    
   # Invert the y operator, take ddy, and apply to tmp, storing in the derivative array
    get_ddy_L_y_inverse_out(ddy, tmp, x, y, 
                            A_nm1_y, B_nm1_y,
                            bdry_hist_ay, bdry_hist_by,
                            dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None

#-------------------------------------------------------------
# Advance for the second y-derivative of the numerical solution
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64[:], nb.float64[:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def BDF2_dyy_advance_out(dyy, v, src_data, x, y, t, 
                         A_nm1_x, B_nm1_x, A_nm1_y, B_nm1_y,
                         bdry_hist_ax, bdry_hist_bx, 
                         bdry_hist_ay, bdry_hist_by,
                         dx, dy, dt, c, beta):
    """
    Calculates the dyy of the solution to the wave equation using the second-order BDF method. 
    This function accepts the mesh data v and src_data.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size    
    
    # Variables for the integrands
    R = np.empty((N_x, N_y)) # time history
    tmp = np.empty((N_x, N_y)) # tmp storage for the inverse
    
    # Time history (v doesn't include the extension region)
    alpha = beta/(c*dt)
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            R[i,j] = 0.5*( 5*v[2,i,j] - 4*v[1,i,j] + v[0,i,j] )
            
            # Contribution from the source term (at t_{n+1})
            R[i,j] += ( 1/(alpha**2) )*src_data[i,j]

    # Invert the x operator and apply to R, storing in tmp
    get_L_x_inverse_out(tmp, R, x, y, 
                        A_nm1_x, B_nm1_x,
                        bdry_hist_ax, bdry_hist_bx,
                        dx, dy, dt, c, beta)
    
   # Invert the y operator, take dyy, and apply to tmp, storing in the derivative array
    get_dyy_L_y_inverse_out(dyy, tmp, x, y, 
                            A_nm1_y, B_nm1_y,
                            bdry_hist_ay, bdry_hist_by,
                            dx, dy, dt, c, beta)
    
    # Shuffle is performed outside this function
    
    return None















#
# Central Methods
#


### Add the central-2 methods here as well... and don't use the C operators 
### for this b/c of the source term. These can be modified later.

### Central-4 outflow needs some work, so we don't include it here...


@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def central2_advance_per(v, src_data, x, y, t,
                         dx, dy, dt, c, beta):
    """
    Calculates a solution to the wave equation using the second-order central method. 
    This function accepts the mesh data v and src_data.
    
    Source function is explicit in the central method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """

    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    beta2 = beta**2
    alpha = beta/(c*dt)
    
    # Temporary storage
    R = np.empty((N_x, N_y))
    tmp1 = np.empty((N_x, N_y))
    tmp2 = np.empty((N_x, N_y))
    
    # Time history (v doesn't include the extension region)
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Define the integrand
            # Contribution from the source term (at t_{n})
            R[i,j] = v[1,i,j] + ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_per(tmp1, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_per(tmp2, tmp1, x, y, dx, dy, dt, c, beta)
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Update the solution at the new time level
            v[2,i,j] = (2-beta2)*v[1,i,j] - v[0,i,j] + beta2*tmp2[i,j]
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def central2_advance_dir(v, src_data, x, y, t,
                         dx, dy, dt, c, beta):
    """
    Calculates a solution to the wave equation using the second-order central method. 
    This function accepts the mesh data v and src_data.
    
    Source function is explicit in the central method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    beta2 = beta**2
    alpha = beta/(c*dt)
    
    # Temporary storage
    R = np.empty((N_x, N_y))
    tmp1 = np.empty((N_x, N_y))
    tmp2 = np.empty((N_x, N_y))
    
    # Time history (v doesn't include the extension region)
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Define the integrand
            # Contribution from the source term (at t_{n})
            R[i,j] = v[1,i,j] + ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_dir(tmp1, R, x, y, dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_dir(tmp2, tmp1, x, y, dx, dy, dt, c, beta)
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Update the solution at the new time level
            v[2,i,j] = (2-beta2)*v[1,i,j] - v[0,i,j] + beta2*tmp2[i,j]
    
    # Shuffle is performed outside this function
    
    return None


@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:], nb.float64,
                  nb.float64[:], nb.float64[:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64,
                  nb.float64, nb.float64)], 
                  parallel=True, cache=False, boundscheck=False)
def central2_advance_out(v, src_data, x, y, t,
                         A_nm1_x, B_nm1_x, A_nm1_y, B_nm1_y,
                         bdry_hist_ax, bdry_hist_bx,
                         bdry_hist_ay, bdry_hist_by,
                         dx, dy, dt, c, beta):
    """
    Calculates a solution to the wave equation using the second-order central method. 
    This function accepts the mesh data v and src_data. The outflow boundaries use
    extrapolation of the integrand.
    
    Source function is explicit in the central method.
    
    Note that all arrays are passed by reference, unless otherwise stated.
    """
    # Build the convolution integral over the time history R
    N_x = x.size
    N_y = y.size
    beta2 = beta**2
    alpha = beta/(c*dt)
    
    # Temporary storage
    R = np.empty((N_x, N_y))
    tmp1 = np.empty((N_x, N_y))
    tmp2 = np.empty((N_x, N_y))
    
    # Time history (v doesn't include the extension region)
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            # Define the integrand
            # Contribution from the source term (at t_{n})
            R[i,j] = v[1,i,j] + ( 1/(alpha**2) )*src_data[i,j]
            
    # Invert the y operator and apply to R, storing in tmp
    get_L_y_inverse_out(tmp1, R, x, y, 
                        A_nm1_y, B_nm1_y,
                        bdry_hist_ay, bdry_hist_by,
                        dx, dy, dt, c, beta)
    
    # Invert the x operator and apply to tmp, then store in the new time level
    get_L_x_inverse_out(tmp2, tmp1, x, y, 
                        A_nm1_x, B_nm1_x,
                        bdry_hist_ax, bdry_hist_bx,
                        dx, dy, dt, c, beta)
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
    
            # Update the solution at the new time level
            v[2,i,j] = (2-beta2)*v[1,i,j] - v[0,i,j] + beta2*tmp2[i,j]
    
    # Shuffle is performed outside this function
    
    return None





