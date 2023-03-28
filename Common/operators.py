import numpy as np
import numba as nb

from extension import *
from quadrature import *
from summation import *
from boundary import *


# #-------------------------------------------------------------
# # Operators for periodic problems
# #-------------------------------------------------------------

# def get_C_xy_per(operand, x, y, dx, dy, dt, c, beta):
    
#     # Note: Extension is done inside the inverse operation
    
#     C_xy = np.zeros_like(operand)
    
#     # First term
#     D_x = get_D_x_per(operand, x, y, dx, dy, dt, c, beta)
#     tmp = get_L_y_inverse_per(D_x, x, y, dx, dy, dt, c, beta)
    
#     # Store the result in C
#     C_xy += tmp
    
#     # Second term
#     D_y = get_D_y_per(operand, x, y, dx, dy, dt, c, beta)
#     tmp = get_L_x_inverse_per(D_y, x, y, dx, dy, dt, c, beta)
    
#     # Add the result in C
#     C_xy += tmp
    
#     return C_xy

# def get_D_x_per(operand, x, y, dx, dy, dt, c, beta):
    
#     # Compute the inverse
#     L_x_inverse = get_L_x_inverse_per(operand, x, y, 
#                                       dx, dy, dt, c, beta)

#     D_x = operand - L_x_inverse
    
#     return D_x

# def get_D_y_per(operand, x, y, dx, dy, dt, c, beta):
    
#     # Compute the inverse
#     L_y_inverse = get_L_y_inverse_per(operand, x, y, 
#                                       dx, dy, dt, c, beta)
    
#     D_y = operand - L_y_inverse
    
#     return D_y

# def get_D_xy_per(operand, x, y, dx, dy, dt, c, beta):
    
#     # Note: Extension is done inside the inverse operation
    
#     # Compute the inverse along y
#     tmp1 = get_L_y_inverse_per(operand, x, y, dx, dy, dt, c, beta)
    
#     # Compute the inverse along x
#     tmp2 = get_L_x_inverse_per(tmp1, x, y, dx, dy, dt, c, beta)
    
#     # Compute D_xy
#     D_xy = operand - tmp2
    
#     return D_xy


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64) ], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_L_x_inverse_per(inverse, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    mu_x = np.exp(-alpha*( x[-1] - x[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along x
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        
        periodic_extension(operand_ext[:,j+2])
    
    # Invert the x operator and apply to the operand
    for j in nb.prange(N_y):

        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)

        # Combine the integrals into the right-moving operator
        # This gives the convolution integral
        rite_moving_op[:,j] += left_moving_op[:,j]
        rite_moving_op[:,j] *= 0.5
        
        I_a = rite_moving_op[0 ,j]
        I_b = rite_moving_op[-1,j]
            
        A_x = I_b/(1-mu_x)
        B_x = I_a/(1-mu_x)
        
        # Sweep the x boundary data into the operator
        apply_A_and_B(rite_moving_op[:,j], x, alpha, A_x, B_x)
        
    # Transfer contents to the inverse array
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            inverse[i,j] = rite_moving_op[i,j]
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64) ], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_L_y_inverse_per(inverse, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    mu_y = np.exp(-alpha*( y[-1] - y[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the integrand along y
    # Corners are not needed
    for i in nb.prange(N_x):
        
        periodic_extension(operand_ext[i+2,:])
            
    # Invert the y operator and apply to the operand
    for i in nb.prange(N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)
        
        # Combine the integrals into the right-moving operator
        # This gives the convolution integral
        rite_moving_op[i,:] += left_moving_op[i,:]
        rite_moving_op[i,:] *= 0.5
        
        I_a = rite_moving_op[i,0]
        I_b = rite_moving_op[i,-1]
        
        A_y = I_b/(1-mu_y) 
        B_y = I_a/(1-mu_y) 
        
        # Sweep the y boundary data into the operator
        apply_A_and_B(rite_moving_op[i,:], y, alpha, A_y, B_y)
        
    # Transfer contents to the inverse array
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            inverse[i,j] = rite_moving_op[i,j]
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64) ], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_ddx_L_x_inverse_per(ddx, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    mu_x = np.exp(-alpha*( x[-1] - x[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along x
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        
        periodic_extension(operand_ext[:,j+2])
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the x-direction
    #==========================================================================
    for j in nb.prange(N_y):

        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)

        # Add the boundary terms to this convolution integral
        # *** assuming we are periodic ***
        A_x = (rite_moving_op[-1,j] + left_moving_op[-1,j])/(2 - 2*mu_x)
        B_x = (rite_moving_op[0,j] + left_moving_op[0,j])/(2 - 2*mu_x)
        
        # Sweep the x boundary data into the operator
        for i in range(N_x):
            
            ddx[i,j] = -0.5*alpha*rite_moving_op[i,j] + 0.5*alpha*left_moving_op[i,j] \
                    - alpha*A_x*np.exp(-alpha*(x[i] - x[0])) \
                    + alpha*B_x*np.exp(-alpha*(x[-1] - x[i]))
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64) ], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_ddy_L_y_inverse_per(ddy, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    mu_y = np.exp(-alpha*( y[-1] - y[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    for i in nb.prange(N_x):
        
        periodic_extension(operand_ext[i+2,:])
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the y-direction
    #==========================================================================
    for i in nb.prange(N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

        # Add the boundary terms to this convolution integral
        # *** assuming we are periodic ***
        A_y = (rite_moving_op[i,-1] + left_moving_op[i,-1])/(2 - 2*mu_y)
        B_y = (rite_moving_op[i,0] + left_moving_op[i,0])/(2 - 2*mu_y)
        
        for j in range(N_y):
            
            ddy[i,j] = -0.5*alpha*rite_moving_op[i,j] + 0.5*alpha*left_moving_op[i,j] \
                    - alpha*A_y*np.exp(-alpha*(y[j] - y[0])) \
                    + alpha*B_y*np.exp(-alpha*(y[-1] - y[j]))
    
    return None

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64) ], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_dxx_L_x_inverse_per(dxx, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    alpha2 = alpha**2
    mu_x = np.exp(-alpha*( x[-1] - x[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along x
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        
        periodic_extension(operand_ext[:,j+2])
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the x-direction
    #==========================================================================
    for j in nb.prange(N_y):

        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)

        # Add the boundary terms to this convolution integral
        # *** assuming we are periodic ***
        A_x = (rite_moving_op[-1,j] + left_moving_op[-1,j])/(2 - 2*mu_x)
        B_x = (rite_moving_op[0,j] + left_moving_op[0,j])/(2 - 2*mu_x)
        
        # Sweep the x boundary data into the operator
        for i in range(N_x):
            
            # Use the analytical expression for the second derivative in x 
            dxx[i,j] = alpha2*( 0.5*(rite_moving_op[i,j] + left_moving_op[i,j]) \
                               - operand[i,j] \
                               + A_x*np.exp(-alpha*(x[i] - x[0])) \
                               + B_x*np.exp(-alpha*(x[-1] - x[i])) )
            
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64) ], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_dyy_L_y_inverse_per(dyy, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    alpha2 = alpha**2
    mu_y = np.exp(-alpha*( y[-1] - y[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    for i in nb.prange(N_x):
        
        periodic_extension(operand_ext[i+2,:])
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the y-direction
    #==========================================================================
    for i in nb.prange(N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

        # Add the boundary terms to this convolution integral
        # *** assuming we are periodic ***
        A_y = (rite_moving_op[i,-1] + left_moving_op[i,-1])/(2 - 2*mu_y)
        B_y = (rite_moving_op[i,0] + left_moving_op[i,0])/(2 - 2*mu_y)
        
        for j in range(N_y):
            
            # Use the analytical expression for the second derivative in y 
            dyy[i,j] = alpha2*( 0.5*(rite_moving_op[i,j] + left_moving_op[i,j]) \
                               - operand[i,j] \
                               + A_y*np.exp(-alpha*(y[j] - y[0])) \
                               + B_y*np.exp(-alpha*(y[-1] - y[j])) )

    return None


# #-------------------------------------------------------------
# # Operators for Dirichlet problems
# #-------------------------------------------------------------

# def get_C_xy_dir(operand, x, y, dx, dy, dt, c, beta):
    
#     # Note: Extension is done inside the inverse operation
    
#     C_xy = np.zeros_like(operand)
    
#     # First term
#     D_x = get_D_x_dir(operand, x, y, dx, dy, dt, c, beta)
#     tmp = get_L_y_inverse_dir(D_x, x, y, dx, dy, dt, c, beta)
    
#     # Store the result in C
#     C_xy += tmp
    
#     # Second term
#     D_y = get_D_y_dir(operand, x, y, dx, dy, dt, c, beta)
#     tmp = get_L_x_inverse_dir(D_y, x, y, dx, dy, dt, c, beta)
    
#     # Add the result in C
#     C_xy += tmp
    
#     return C_xy

# def get_D_x_dir(operand, x, y, dx, dy, dt, c, beta):
    
#     # Compute the inverse
#     L_x_inverse = get_L_x_inverse_dir(operand, x, y, 
#                                       dx, dy, dt, c, beta)

#     D_x = operand - L_x_inverse
    
#     return D_x

# def get_D_y_dir(operand, x, y, dx, dy, dt, c, beta):
    
#     # Compute the inverse
#     L_y_inverse = get_L_y_inverse_dir(operand, x, y, 
#                                       dx, dy, dt, c, beta)
    
#     D_y = operand - L_y_inverse
    
#     return D_y

# def get_D_xy_dir(operand, x, y, dx, dy, dt, c, beta):
    
#     # Note: Extension is done inside the inverse operation
    
#     # Compute the inverse along y
#     tmp1 = get_L_y_inverse_dir(operand, x, y, dx, dy, dt, c, beta)
    
#     # Compute the inverse along x
#     tmp2 = get_L_x_inverse_dir(tmp1, x, y, dx, dy, dt, c, beta)
    
#     # Compute D_xy
#     D_xy = operand - tmp2
    
#     return D_xy


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64) ], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_L_x_inverse_dir(inverse, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    mu_x = np.exp(-alpha*( x[-1] - x[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        
        polynomial_extension(operand_ext[:,j+2])
    
    # Invert the x operator and apply to the operand
    for j in nb.prange(N_y):

        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)

        # Combine the integrals into the right-moving operator
        # This gives the convolution integral
        rite_moving_op[:,j] += left_moving_op[:,j]
        rite_moving_op[:,j] *= 0.5
        
        I_a = rite_moving_op[0 ,j]
        I_b = rite_moving_op[-1,j]
        
        # Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a
        w_b_dir = I_b
        
        A_x = -( w_a_dir - mu_x*w_b_dir )/(1 - mu_x**2)
        B_x = -( w_b_dir - mu_x*w_a_dir )/(1 - mu_x**2)
        
        # Sweep the x boundary data into the operator
        apply_A_and_B(rite_moving_op[:,j], x, alpha, A_x, B_x)
        
    # Transfer contents to the inverse array
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            inverse[i,j] = rite_moving_op[i,j]
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64) ], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_L_y_inverse_dir(inverse, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    mu_y = np.exp(-alpha*( y[-1] - y[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the integrand along y
    # Corners are not needed
    for i in nb.prange(N_x):
        
        polynomial_extension(operand_ext[i+2,:])
            
    # Invert the y operator and apply to the operand
    for i in nb.prange(N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

        # Combine the integrals into the right-moving operator
        # This gives the convolution integral
        rite_moving_op[i,:] += left_moving_op[i,:]
        rite_moving_op[i,:] *= 0.5
        
        I_a = rite_moving_op[i,0]
        I_b = rite_moving_op[i,-1]
        
        # Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a
        w_b_dir = I_b
        
        A_y = -( w_a_dir - mu_y*w_b_dir )/(1 - mu_y**2)
        B_y = -( w_b_dir - mu_y*w_a_dir )/(1 - mu_y**2) 
        
        # Sweep the y boundary data into the operator
        apply_A_and_B(rite_moving_op[i,:], y, alpha, A_y, B_y)
        
    # Transfer contents to the inverse array
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            inverse[i,j] = rite_moving_op[i,j]
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64)], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_ddx_L_x_inverse_dir(ddx, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    mu_x = np.exp(-alpha*( x[-1] - x[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        
        polynomial_extension(operand_ext[:,j+2])
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the x-direction
    #==========================================================================
    for j in nb.prange(N_y):

        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)
        
        # Get the A and B values for Dirichlet
        # Assumes both ends of the line use Dirichlet
        #
        # See the paper "METHOD OF LINES TRANSPOSE: AN EFFICIENT A-STABLE SOLVER FOR WAVE PROPAGATION"
        # By Causley, et al. 2015
        
        I_a = 0.5*( rite_moving_op[0,j] + left_moving_op[0,j] )
        I_b = 0.5*( rite_moving_op[-1,j] + left_moving_op[-1,j] )
        
        # w_a_dir = I_a - u_along_xa(t) - ( u_along_xa(t+dt) - 2*u_along_xa(t) + u_along_xa(t-dt) )/(beta**2)
        # w_b_dir = I_b - u_along_xb(t) - ( u_along_xb(t+dt) - 2*u_along_xb(t) + u_along_xb(t-dt) )/(beta**2)

        # Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a 
        w_b_dir = I_b 
        
        A_x = -( w_a_dir - mu_x*w_b_dir )/(1 - mu_x**2)
        B_x = -( w_b_dir - mu_x*w_a_dir )/(1 - mu_x**2)
        
        # Sweep the x boundary data into the operator
        for i in range(N_x):
            
            ddx[i,j] = -0.5*alpha*rite_moving_op[i,j] + 0.5*alpha*left_moving_op[i,j] \
                    - alpha*A_x*np.exp(-alpha*(x[i] - x[0])) \
                    + alpha*B_x*np.exp(-alpha*(x[-1] - x[i]))
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64)], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_ddy_L_y_inverse_dir(ddy, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    mu_y = np.exp(-alpha*( y[-1] - y[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the integrand along y
    # Corners are not needed
    for i in nb.prange(N_x):
        
        polynomial_extension(operand_ext[i+2,:])
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the y-direction
    #==========================================================================
    for i in nb.prange(N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

        # Get the A and B values for Dirichlet
        # Assumes both ends of the line use Dirichlet
        #
        # See the paper "METHOD OF LINES TRANSPOSE: AN EFFICIENT A-STABLE SOLVER FOR WAVE PROPAGATION"
        # By Causley, et al. 2015
        I_a = 0.5*( rite_moving_op[i,0] + left_moving_op[i,0] )
        I_b = 0.5*( rite_moving_op[i,-1] + left_moving_op[i,-1] )
        
        # w_a_dir = I_a - u_along_xa(t) - ( u_along_xa(t+dt) - 2*u_along_xa(t) + u_along_xa(t-dt) )/(beta**2)
        # w_b_dir = I_b - u_along_xb(t) - ( u_along_xb(t+dt) - 2*u_along_xb(t) + u_along_xb(t-dt) )/(beta**2)

        # Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a 
        w_b_dir = I_b 
        
        A_y = -( w_a_dir - mu_y*w_b_dir )/(1 - mu_y**2)
        B_y = -( w_b_dir - mu_y*w_a_dir )/(1 - mu_y**2)
        
        for j in range(N_y):
            
            ddy[i,j] = -0.5*alpha*rite_moving_op[i,j] + 0.5*alpha*left_moving_op[i,j] \
                    - alpha*A_y*np.exp(-alpha*(y[j] - y[0])) \
                    + alpha*B_y*np.exp(-alpha*(y[-1] - y[j]))
    
    return None

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64)], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_dxx_L_x_inverse_dir(dxx, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    alpha2 = alpha**2
    mu_x = np.exp(-alpha*( x[-1] - x[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        
        polynomial_extension(operand_ext[:,j+2])
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the x-direction
    #==========================================================================
    for j in nb.prange(N_y):

        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)
        
        # Get the A and B values for Dirichlet
        # Assumes both ends of the line use Dirichlet
        #
        # See the paper "METHOD OF LINES TRANSPOSE: AN EFFICIENT A-STABLE SOLVER FOR WAVE PROPAGATION"
        # By Causley, et al. 2015
        
        I_a = 0.5*( rite_moving_op[0,j] + left_moving_op[0,j] )
        I_b = 0.5*( rite_moving_op[-1,j] + left_moving_op[-1,j] )

        # Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a 
        w_b_dir = I_b 
        
        A_x = -( w_a_dir - mu_x*w_b_dir )/(1 - mu_x**2)
        B_x = -( w_b_dir - mu_x*w_a_dir )/(1 - mu_x**2)
        
        # Sweep the x boundary data into the operator
        for i in range(N_x):
            
            # Use the analytical expression for the second derivative in x 
            dxx[i,j] = alpha2*( 0.5*(rite_moving_op[i,j] + left_moving_op[i,j]) \
                               - operand[i,j] \
                               + A_x*np.exp(-alpha*(x[i] - x[0])) \
                               + B_x*np.exp(-alpha*(x[-1] - x[i])) )
            
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:],
                          nb.float64, nb.float64, nb.float64, 
                          nb.float64, nb.float64)], 
                          parallel=True, nogil=True,
                          cache=False, boundscheck=False)
def get_dyy_L_y_inverse_dir(dyy, operand, x, y, dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    alpha2 = alpha**2
    mu_y = np.exp(-alpha*( y[-1] - y[0] ) )           
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the integrand along y
    # Corners are not needed
    for i in nb.prange(N_x):
        
        polynomial_extension(operand_ext[i+2,:])
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the y-direction
    #==========================================================================
    for i in nb.prange(N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

        # Get the A and B values for Dirichlet
        # Assumes both ends of the line use Dirichlet
        #
        # See the paper "METHOD OF LINES TRANSPOSE: AN EFFICIENT A-STABLE SOLVER FOR WAVE PROPAGATION"
        # By Causley, et al. 2015
        I_a = 0.5*( rite_moving_op[i,0] + left_moving_op[i,0] )
        I_b = 0.5*( rite_moving_op[i,-1] + left_moving_op[i,-1] )
        
        # Assume homogeneous BCs (same expression for BDF and central schemes)
        w_a_dir = I_a 
        w_b_dir = I_b 
        
        A_y = -( w_a_dir - mu_y*w_b_dir )/(1 - mu_y**2)
        B_y = -( w_b_dir - mu_y*w_a_dir )/(1 - mu_y**2)
        
        for j in range(N_y):
            
            # Use the analytical expression for the second derivative in y 
            dyy[i,j] = alpha2*( 0.5*(rite_moving_op[i,j] + left_moving_op[i,j]) \
                               - operand[i,j] \
                               + A_y*np.exp(-alpha*(y[j] - y[0])) \
                               + B_y*np.exp(-alpha*(y[-1] - y[j])) )
    
    return None


#-------------------------------------------------------------
# Operators for outflow problems
#-------------------------------------------------------------

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64, 
                  nb.float64, nb.float64)], 
                  parallel=True, nogil=True, 
                  cache=False, boundscheck=False)
def get_L_x_inverse_out(inverse, operand, x, y, 
                        A_nm1_x, B_nm1_x,
                        bdry_hist_ax, bdry_hist_bx,
                        dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt) 

    # Note: We use integration weights for the central-2
    # scheme. We store the history of the integrand along
    # the boundary and use this to approximate the
    # outflow conditions.
    gamma = get_explicit_central2_outflow_wts(beta)
        
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        polynomial_extension(operand_ext[:,j+2])
        
    # Update the boundary history at time level n for the operand
    for j in nb.prange(N_y): 
        bdry_hist_ax[j,-1] = operand[ 0,j]
        bdry_hist_bx[j,-1] = operand[-1,j]
    
    # Invert the x operator and apply to the operand
    for j in nb.prange(0,N_y):

        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        
        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)
        
        # Combine the integrals into the right-moving operator
        # This gives the convolution integral
        rite_moving_op[:,j] += left_moving_op[:,j]
        rite_moving_op[:,j] *= 0.5
        
        # Compute the outflow coefficients
        
        # Extract the A and B values from the previous time level
        A_nm1 = A_nm1_x[j]
        B_nm1 = B_nm1_x[j]
            
        # Slice for the boundary history along this line
        # Here we need the last 3 time points
        # Input slice already excludes n+1
        bdry_hist_a = bdry_hist_ax[j,-3:]
        bdry_hist_b = bdry_hist_bx[j,-3:]

        # Use the explicit form of the outflow method
        A_xn = np.exp(-beta)*A_nm1 + gamma[0]*bdry_hist_a[-1] + gamma[1]*bdry_hist_a[-2] + gamma[2]*bdry_hist_a[-3]

        B_xn = np.exp(-beta)*B_nm1 + gamma[0]*bdry_hist_b[-1] + gamma[1]*bdry_hist_b[-2] + gamma[2]*bdry_hist_b[-3]
        
        # Sweep the x boundary data into the operator
        apply_A_and_B(rite_moving_op[:,j], x, alpha, A_xn, B_xn)
        
        # Store the boundary data for the next step
        A_nm1_x[j] = A_xn
        B_nm1_x[j] = B_xn
        
    # Transfer contents to the inverse array
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            inverse[i,j] = rite_moving_op[i,j]
    
    return None

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64, 
                  nb.float64, nb.float64)], 
                  parallel=True, nogil=True, 
                  cache=False, boundscheck=False)
def get_L_y_inverse_out(inverse, operand, x, y,
                        A_nm1_y, B_nm1_y,
                        bdry_hist_ay, bdry_hist_by,
                        dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
        
    # Note: We use integration weights for the central-2
    # scheme. We store the history of the integrand along
    # the boundary and use this to approximate the
    # outflow conditions.
    gamma = get_explicit_central2_outflow_wts(beta)
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the integrand along y
    # Corners are not needed
    for i in nb.prange(N_x):
        polynomial_extension(operand_ext[i+2,:])
        
    # Update the boundary history at time level n for the operand
    for i in nb.prange(N_x):
        bdry_hist_ay[i,-1] = operand[i, 0]
        bdry_hist_by[i,-1] = operand[i,-1]
            
    # Invert the y operator and apply to the operand
    for i in nb.prange(0,N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        # In the centered scheme, the integral is on u^{n} data
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

        # Combine the integrals into the right-moving operator
        # This gives the convolution integral
        rite_moving_op[i,:] += left_moving_op[i,:]
        rite_moving_op[i,:] *= 0.5

        # Compute the outflow coefficients
        
        # Extract the A and B values from the previous time level
        A_nm1 = A_nm1_y[i]
        B_nm1 = B_nm1_y[i]
            
        # Slice for the boundary history along this line
        # Here we need the last 3 time points
        # Input slice already excludes n+1
        bdry_hist_a = bdry_hist_ay[i,-3:]
        bdry_hist_b = bdry_hist_by[i,-3:]
        
        # Use the explicit form of the outflow method
        A_yn = np.exp(-beta)*A_nm1 + gamma[0]*bdry_hist_a[-1] + gamma[1]*bdry_hist_a[-2] + gamma[2]*bdry_hist_a[-3]

        B_yn = np.exp(-beta)*B_nm1 + gamma[0]*bdry_hist_b[-1] + gamma[1]*bdry_hist_b[-2] + gamma[2]*bdry_hist_b[-3]
        
        # Sweep the y boundary data into the operator
        apply_A_and_B(rite_moving_op[i,:], y, alpha, A_yn, B_yn)
        
        # Store the boundary data for the next step
        A_nm1_y[i] = A_yn
        B_nm1_y[i] = B_yn
        
    # Transfer contents to the inverse array
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            inverse[i,j] = rite_moving_op[i,j]
    
    return None


# #-----------------------------------------------------------------
# # Specializations based on central/successive convolution schemes
# #
# # Are these needed???
# #-----------------------------------------------------------------

# @nb.njit([nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.float64[:],
#                           nb.float64[:], nb.float64[:],
#                           nb.float64[:,:], nb.float64[:,:],
#                           nb.float64, nb.float64, nb.float64, 
#                           nb.float64, nb.float64, nb.int64)], 
#                           cache = True, boundscheck = True)
# def get_L_x_inverse_out_C(operand, x, y, 
#                           A_nm1_x, B_nm1_x,
#                           v_prev_ax, v_prev_bx,
#                           dx, dy, dt, c, beta, level = 1):
    
#     N_x = x.size
#     N_y = y.size
    
#     alpha = beta/(c*dt)
#     mu_x = np.exp(-alpha*( x[-1] - x[0] ) ) 

#     # Compute the integration weights for the outflow condition based on the level
#     if level == 1:
#         Gamma = get_implicit_central2_outflow_wts(beta)
#     else:
#         Gamma = get_explicit_central4_outflow_wts(beta)
        
#     # Create arrays for right and left-moving operators
#     rite_moving_op = np.zeros((N_x, N_y))
#     left_moving_op = np.zeros((N_x, N_y))
    
#     # Extend the data for the integrand along y
#     # Corners are not needed
#     operand_ext = np.zeros((N_x+4, N_y+4))
    
#     # Transfer the mesh data
#     operand_ext[2:-2,2:-2] = operand[:,:]
    
#     # Extend the data for the operand along x
#     # Corners are not needed
#     for j in range(N_y):
#         polynomial_extension(operand_ext[:,j+2])
    
#     # Invert the x operator and apply to the operand
#     for j in range(1,N_y-1):

#         # Get the local integrals
#         # Note that function names are reversed, as we use the old convention
#         linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
#         linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

#         # FC step
#         fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)

#         # Combine the integrals into the right-moving operator
#         # This gives the convolution integral
#         rite_moving_op[:,j] += left_moving_op[:,j]
#         rite_moving_op[:,j] *= 0.5
        
#         I_a = rite_moving_op[0 ,j]
#         I_b = rite_moving_op[-1,j]
        
#         A_nm1 = A_nm1_x[j]
#         B_nm1 = B_nm1_x[j]
        
#         # Compute the outflow coefficients based on the level
#         # Level construct applies only to the central scheme...
#         if level == 1:
            
#             # Slice for the boundary history along this line
#             # Here we only need the last 2 time points
#             # Input slice already excludes n+1
#             v_prev_a = v_prev_ax[j,-2:]
#             v_prev_b = v_prev_bx[j,-2:]
            
#             # Use the implicit recurrence
#             A_xn, B_xn = apply_level1_out(I_a, I_b, v_prev_a, v_prev_b, 
#                                           A_nm1, B_nm1, Gamma, mu_x, beta)
            
#         else:
            
#             # Slice for the boundary history along this line
#             # Here we need the last 5 time points
#             # Input slice already excludes n+1
#             v_prev_a = v_prev_ax[j,-5:]
#             v_prev_b = v_prev_bx[j,-5:]
            
#             # Use the explicit recurrence
#             A_xn, B_xn = apply_level2_out(v_prev_a, v_prev_b, 
#                                           A_nm1, B_nm1, Gamma, beta)
        
#         # Sweep the x boundary data into the operator
#         apply_A_and_B(rite_moving_op[:,j], x, alpha, A_xn, B_xn)
        
#         # Store the boundary data for the next step
#         A_nm1_x[j] = A_xn
#         B_nm1_x[j] = B_xn
    
#     return rite_moving_op


# @nb.njit([nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.float64[:],
#                           nb.float64[:], nb.float64[:],
#                           nb.float64[:,:], nb.float64[:,:],
#                           nb.float64, nb.float64, nb.float64, 
#                           nb.float64, nb.float64, nb.int64)], 
#                           cache = True, boundscheck = True)
# def get_L_y_inverse_out_C(operand, x, y,
#                           A_nm1_y, B_nm1_y,
#                           v_prev_ay, v_prev_by,
#                           dx, dy, dt, c, beta, 
#                           level = 1):
    
#     N_x = x.size
#     N_y = y.size
    
#     alpha = beta/(c*dt)
#     mu_y = np.exp(-alpha*( y[-1] - y[0] ) )
        
#     # Compute the integration weights for the outflow condition based on the level
#     if level == 1:
#         Gamma = get_implicit_central2_outflow_wts(beta)
#     else:
#         Gamma = get_explicit_central4_outflow_wts(beta)
    
#     # Create arrays for right and left-moving operators
#     rite_moving_op = np.zeros((N_x, N_y))
#     left_moving_op = np.zeros((N_x, N_y))
    
#     # Extend the data for the integrand along y
#     # Corners are not needed
#     operand_ext = np.zeros((N_x+4, N_y+4))
    
#     # Transfer the mesh data
#     operand_ext[2:-2,2:-2] = operand[:,:]
    
#     # Extend the data for the integrand along y
#     # Corners are not needed
#     for i in range(N_x):
#         polynomial_extension(operand_ext[i+2,:])
            
#     # Invert the y operator and apply to the operand
#     for i in range(1,N_x-1):
        
#         # Get the local integrals 
#         # Note that function names are reversed, as we use the old convention
#         # In the centered scheme, the integral is on u^{n} data
#         linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
#         linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
#         # FC step in for y operator
#         fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

#         # Combine the integrals into the right-moving operator
#         # This gives the convolution integral
#         rite_moving_op[i,:] += left_moving_op[i,:]
#         rite_moving_op[i,:] *= 0.5
        
#         I_a = rite_moving_op[i,0]
#         I_b = rite_moving_op[i,-1]
        
#         A_nm1 = A_nm1_y[i]
#         B_nm1 = B_nm1_y[i]
        
#         # Compute the outflow coefficients based on the level
#         # Level construct applies only to the central scheme...
#         if level == 1:
            
#             # Slice for the boundary history along this line
#             # Here we only need the last 2 time points
#             # Input slice already excludes n+1
#             v_prev_a = v_prev_ay[i,-2:]
#             v_prev_b = v_prev_by[i,-2:]
            
#             # Use the implicit recurrence
#             A_yn, B_yn = apply_level1_out(I_a, I_b, v_prev_a, v_prev_b, 
#                                           A_nm1, B_nm1, Gamma, mu_y, beta)
            
#         else:
            
#             # Slice for the boundary history along this line
#             # Here we need the last 5 time points
#             # Input slice already excludes n+1
#             v_prev_a = v_prev_ay[i,-5:]
#             v_prev_b = v_prev_by[i,-5:]
            
#             # Use the explicit recurrence
#             A_yn, B_yn = apply_level2_out(v_prev_a, v_prev_b, 
#                                           A_nm1, B_nm1, Gamma, beta) 
        
#         # Sweep the y boundary data into the operator
#         apply_A_and_B(rite_moving_op[i,:], y, alpha, A_yn, B_yn)
        
#         # Store the boundary data for the next step
#         A_nm1_y[i] = A_yn
#         B_nm1_y[i] = B_yn
    
#     return rite_moving_op


### To-do: What if we now want to use this function with the BDF-3 or BDF-4 integrand?
### The weights will change, so this will be limited to the second order scheme...
###
### Actually, if we use the central-2 outflow weights in explicit form, we can use any
### of the inverse functions for outflow. The only thing that changes between applications
### is the interpretation of the integrand. So, this means we likely won't need any of
### the specializations for successive convolution methods.

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64, 
                  nb.float64, nb.float64)], 
                  parallel=True, nogil=True, 
                  cache=False, boundscheck=False)
def get_ddx_L_x_inverse_out(ddx_inverse, operand, x, y, 
                            A_nm1_x, B_nm1_x,
                            bdry_hist_ax, bdry_hist_bx,
                            dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)

    # Note: We use integration weights for the central-2
    # scheme. We store the history of the integrand along
    # the boundary and use this to approximate the
    # outflow conditions.
    gamma = get_explicit_central2_outflow_wts(beta)
        
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Array for the derivative
    ddx = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        polynomial_extension(operand_ext[:,j+2])
        
    # Update the boundary history at time level n for the operand
    for j in nb.prange(N_y):
        bdry_hist_ax[j,-1] = operand[ 0,j]
        bdry_hist_bx[j,-1] = operand[-1,j]
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the x-direction
    #==========================================================================
    for j in nb.prange(0,N_y):
    
        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)
        
        # Compute the outflow coefficients
        
        # Extract the A and B values from the previous time level
        A_nm1 = A_nm1_x[j]
        B_nm1 = B_nm1_x[j]
            
        # Slice for the boundary history along this line
        # Here we need the last 3 time points
        # Input slice already excludes n+1
        bdry_hist_a = bdry_hist_ax[j,-3:]
        bdry_hist_b = bdry_hist_bx[j,-3:]

        # Use the explicit form of the outflow method
        A_xn = np.exp(-beta)*A_nm1 + gamma[0]*bdry_hist_a[-1] + gamma[1]*bdry_hist_a[-2] + gamma[2]*bdry_hist_a[-3]

        B_xn = np.exp(-beta)*B_nm1 + gamma[0]*bdry_hist_b[-1] + gamma[1]*bdry_hist_b[-2] + gamma[2]*bdry_hist_b[-3]
        
        # Store the boundary data for the next step
        A_nm1_x[j] = A_xn
        B_nm1_x[j] = B_xn
        
        # Sweep the x boundary data into the operator
        for i in range(N_x):
            
            ddx_inverse[i,j] = -0.5*alpha*rite_moving_op[i,j] + 0.5*alpha*left_moving_op[i,j] \
                    - alpha*A_xn*np.exp(-alpha*(x[i] - x[0])) \
                    + alpha*B_xn*np.exp(-alpha*(x[-1] - x[i]))
    
    return None

### To-do: What if we now want to use this function with the BDF-3 or BDF-4 integrand?
### The weights will change, so this will be limited to the second order scheme...
###
### Actually, if we use the central-2 outflow weights in explicit form, we can use any
### of the inverse functions for outflow. The only thing that changes between applications
### is the interpretation of the integrand. So, this means we likely won't need any of
### the specializations for successive convolution methods.

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64, 
                  nb.float64, nb.float64)], 
                  parallel=True, nogil=True, 
                  cache=False, boundscheck=False)
def get_ddy_L_y_inverse_out(ddy_inverse, operand, x, y,
                            A_nm1_y, B_nm1_y,
                            bdry_hist_ay, bdry_hist_by,
                            dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)

    # Note: We use integration weights for the central-2
    # scheme. We store the history of the integrand along
    # the boundary and use this to approximate the
    # outflow conditions.
    gamma = get_explicit_central2_outflow_wts(beta)
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Array for the derivative
    ddy = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the integrand along y
    # Corners are not needed
    for i in nb.prange(N_x):
        polynomial_extension(operand_ext[i+2,:])
        
    # Update the boundary history at time level n for the operand
    for i in nb.prange(N_x):
        bdry_hist_ay[i,-1] = operand[i, 0]
        bdry_hist_by[i,-1] = operand[i,-1]
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the y-direction
    #==========================================================================
    for i in nb.prange(0,N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        # In the centered scheme, the integral is on u^{n} data
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

        # Compute the outflow coefficients
        
        # Extract the A and B values from the previous time level
        A_nm1 = A_nm1_y[i]
        B_nm1 = B_nm1_y[i]
            
        # Slice for the boundary history along this line
        # Here we need the last 3 time points
        # Input slice already excludes n+1
        bdry_hist_a = bdry_hist_ay[i,-3:]
        bdry_hist_b = bdry_hist_by[i,-3:]
        
        # Use the explicit form of the outflow method
        A_yn = np.exp(-beta)*A_nm1 + gamma[0]*bdry_hist_a[-1] + gamma[1]*bdry_hist_a[-2] + gamma[2]*bdry_hist_a[-3]

        B_yn = np.exp(-beta)*B_nm1 + gamma[0]*bdry_hist_b[-1] + gamma[1]*bdry_hist_b[-2] + gamma[2]*bdry_hist_b[-3]
        
        # Store the boundary data for the next step
        A_nm1_y[i] = A_yn
        B_nm1_y[i] = B_yn
        
        # Sweep the x boundary data into the operator
        for j in range(N_y):
            
            ddy_inverse[i,j] = -0.5*alpha*rite_moving_op[i,j] + 0.5*alpha*left_moving_op[i,j] \
                    - alpha*A_yn*np.exp(-alpha*(y[j] - y[0])) \
                    + alpha*B_yn*np.exp(-alpha*(y[-1] - y[j]))
    
    return None


@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64, 
                  nb.float64, nb.float64)], 
                  parallel=True, nogil=True, 
                  cache=False, boundscheck=False)
def get_dxx_L_x_inverse_out(dxx_inverse, operand, x, y, 
                            A_nm1_x, B_nm1_x,
                            bdry_hist_ax, bdry_hist_bx,
                            dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    alpha2 = alpha**2

    # Note: We use integration weights for the central-2
    # scheme. We store the history of the integrand along
    # the boundary and use this to approximate the
    # outflow conditions.
    gamma = get_explicit_central2_outflow_wts(beta)
        
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Array for the derivative
    ddx = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    # Transfer the mesh data
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the operand along x
    # Corners are not needed
    for j in nb.prange(N_y):
        polynomial_extension(operand_ext[:,j+2])
        
    # Update the boundary history at time level n for the operand
    for j in nb.prange(N_y):
        bdry_hist_ax[j,-1] = operand[ 0,j]
        bdry_hist_bx[j,-1] = operand[-1,j]
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the x-direction
    #==========================================================================
    for j in nb.prange(0,N_y):
    
        # Get the local integrals
        # Note that function names are reversed, as we use the old convention
        linear5_L(rite_moving_op[:,j], operand_ext[:,j+2], alpha, dx)
        linear5_R(left_moving_op[:,j], operand_ext[:,j+2], alpha, dx)

        # FC step
        fast_convolution(rite_moving_op[:,j], left_moving_op[:,j], alpha, dx)
        
        # Compute the outflow coefficients
        
        # Extract the A and B values from the previous time level
        A_nm1 = A_nm1_x[j]
        B_nm1 = B_nm1_x[j]
            
        # Slice for the boundary history along this line
        # Here we need the last 3 time points
        # Input slice already excludes n+1
        bdry_hist_a = bdry_hist_ax[j,-3:]
        bdry_hist_b = bdry_hist_bx[j,-3:]

        # Use the explicit form of the outflow method
        A_xn = np.exp(-beta)*A_nm1 + gamma[0]*bdry_hist_a[-1] + gamma[1]*bdry_hist_a[-2] + gamma[2]*bdry_hist_a[-3]

        B_xn = np.exp(-beta)*B_nm1 + gamma[0]*bdry_hist_b[-1] + gamma[1]*bdry_hist_b[-2] + gamma[2]*bdry_hist_b[-3]
        
        # Store the boundary data for the next step
        A_nm1_x[j] = A_xn
        B_nm1_x[j] = B_xn
        
        # Sweep the x boundary data into the operator
        for i in range(N_x):
            
            # Use the analytical expression for the second derivative in x 
            dxx_inverse[i,j] = alpha2*( 0.5*(rite_moving_op[i,j] + left_moving_op[i,j]) \
                               - operand[i,j] \
                               + A_xn*np.exp(-alpha*(x[i] - x[0])) \
                               + B_xn*np.exp(-alpha*(x[-1] - x[i])) )

    return None

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], 
                  nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:],
                  nb.float64[:,:], nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64, 
                  nb.float64, nb.float64)], 
                  parallel=True, nogil=True, 
                  cache=False, boundscheck=False)
def get_dyy_L_y_inverse_out(dyy_inverse, operand, x, y,
                            A_nm1_y, B_nm1_y,
                            bdry_hist_ay, bdry_hist_by,
                            dx, dy, dt, c, beta):
    
    N_x = x.size
    N_y = y.size
    
    alpha = beta/(c*dt)
    alpha2 = alpha**2
    
    # Note: We use integration weights for the central-2
    # scheme. We store the history of the integrand along
    # the boundary and use this to approximate the
    # outflow conditions.
    gamma = get_explicit_central2_outflow_wts(beta)
    
    # Create arrays for right and left-moving operators
    rite_moving_op = np.zeros((N_x, N_y))
    left_moving_op = np.zeros((N_x, N_y))
    
    # Array for the derivative
    ddy = np.zeros((N_x, N_y))
    
    # Extend the data for the integrand along y
    # Corners are not needed
    operand_ext = np.zeros((N_x+4, N_y+4))
    
    for i in nb.prange(N_x):
        for j in nb.prange(N_y):
            operand_ext[i+2,j+2] = operand[i,j]
    
    # Extend the data for the integrand along y
    # Corners are not needed
    for i in nb.prange(N_x):
        polynomial_extension(operand_ext[i+2,:])
        
    # Update the boundary history at time level n for the operand
    for i in nb.prange(N_x):
        bdry_hist_ay[i,-1] = operand[i, 0]
        bdry_hist_by[i,-1] = operand[i,-1]
    
    #==========================================================================
    # Invert the 1-D Helmholtz operator in the y-direction
    #==========================================================================
    for i in nb.prange(0,N_x):
        
        # Get the local integrals 
        # Note that function names are reversed, as we use the old convention
        # In the centered scheme, the integral is on u^{n} data
        linear5_L(rite_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        linear5_R(left_moving_op[i,:], operand_ext[i+2,:], alpha, dy)
        
        # FC step in for y operator
        fast_convolution(rite_moving_op[i,:], left_moving_op[i,:], alpha, dy)

        # Compute the outflow coefficients
        
        # Extract the A and B values from the previous time level
        A_nm1 = A_nm1_y[i]
        B_nm1 = B_nm1_y[i]
            
        # Slice for the boundary history along this line
        # Here we need the last 3 time points
        # Input slice already excludes n+1
        bdry_hist_a = bdry_hist_ay[i,-3:]
        bdry_hist_b = bdry_hist_by[i,-3:]
        
        # Use the explicit form of the outflow method
        A_yn = np.exp(-beta)*A_nm1 + gamma[0]*bdry_hist_a[-1] + gamma[1]*bdry_hist_a[-2] + gamma[2]*bdry_hist_a[-3]

        B_yn = np.exp(-beta)*B_nm1 + gamma[0]*bdry_hist_b[-1] + gamma[1]*bdry_hist_b[-2] + gamma[2]*bdry_hist_b[-3]
        
        # Store the boundary data for the next step
        A_nm1_y[i] = A_yn
        B_nm1_y[i] = B_yn
        
        # Sweep the x boundary data into the operator
        for j in range(N_y):
                    
            # Use the analytical expression for the second derivative in y 
            dyy_inverse[i,j] = alpha2*( 0.5*(rite_moving_op[i,j] + left_moving_op[i,j]) \
                               - operand[i,j] \
                               + A_yn*np.exp(-alpha*(y[j] - y[0])) \
                               + B_yn*np.exp(-alpha*(y[-1] - y[j])) )
    
    return None









