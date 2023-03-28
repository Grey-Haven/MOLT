import numpy as np
import numba as nb

@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                  nb.float64, nb.float64)], 
                  cache = False, boundscheck = False)
def fixed3_L(J_L, v, alpha, dx):
    '''
    Compute the second order accurate approximation to the 
    left convolution integral using a three point stencil. This
    version does NOT use an extension regions.
    
    Note: We use the old convention, where L means right-moving op
    '''
    
    N = v.size
    nu = alpha*dx
    nu2 = nu*nu
    e_mnu = np.exp(-nu)
    
    #----------------------------------------------------------------------------------------------------
    # Compute the local integrals J_{i}^{L} on x_{i-1} to x_{i}, i = 1,...,N+1
    #----------------------------------------------------------------------------------------------------

    J_L[0] = 0.0
    
    # Interior stencil coefficients
    cl_im1 = -(2*e_mnu - nu + 2*nu2*e_mnu + 3*nu*e_mnu - 2)/(2*nu2)
    cl_i = (2*e_mnu + nu2 + 2*nu*e_mnu - 2)/nu2
    cl_ip1 = -(2*e_mnu + nu + nu*e_mnu - 2)/(2*nu2)
    
    # Note, we stop one before the end and use a biased stencil
    for i in range(1,N-1):
        
        J_L[i] = cl_im1*v[i-1] + cl_i*v[i] + cl_ip1*v[i+1]
        
    # Left-biased stencil for the final point
    cl_m3 = -(2*e_mnu + nu + nu*e_mnu - 2)/(2*nu2)
    cl_m2 = (2*e_mnu + 2*nu - nu2*e_mnu - 2)/nu2
    cl_m1 = (2*nu2 - 3*nu - 2*e_mnu + nu*e_mnu + 2)/(2*nu2)
    
    J_L[-1] = cl_m3*v[-3] + cl_m2*v[-2] + cl_m1*v[-1]
    
    return None
    

@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                  nb.float64, nb.float64)], 
                  cache = False, boundscheck = False)
def fixed3_R(J_R, v, alpha, dx):
    '''
    Compute the second order accurate approximation to the 
    right convolution integral using a three point stencil. This
    version does NOT use an extension regions.
    
    Note: We use the old convention, where R means left-moving op
    '''
    
    N = v.size
    nu = alpha*dx
    nu2 = nu*nu
    e_mnu = np.exp(-nu)
    
    #----------------------------------------------------------------------------------------------------
    # Compute the local integrals J_{i}^{R} on x_{i} to x_{i+1}, i = 0,...,N
    #----------------------------------------------------------------------------------------------------

    # Right-biased stencil for the first point
    cr_0 = (2*nu2 - 3*nu - 2*e_mnu + nu*e_mnu + 2)/(2*nu2)
    cr_1 = (2*e_mnu + 2*nu - nu2*e_mnu - 2)/nu2
    cr_2 = -(2*e_mnu + nu + nu*e_mnu - 2)/(2*nu2)
    
    J_R[0] = cr_0*v[0] + cr_1*v[1] + cr_2*v[2]

    # Interior stencil coefficients
    cr_im1 = -(2*e_mnu + nu + nu*e_mnu - 2)/(2*nu2)
    cr_i = (2*e_mnu + nu2 + 2*nu*e_mnu - 2)/nu2
    cr_ip1 = -(2*e_mnu - nu + 2*nu2*e_mnu + 3*nu*e_mnu - 2)/(2*nu2)
    
    # Note, we start one element to the right and use a biased stencil
    for i in range(1,N-1):
        
        J_R[i] = cr_im1*v[i-1] + cr_i*v[i] + cr_ip1*v[i+1]
        
    J_R[-1] = 0.0
    
    return None


@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                  nb.float64, nb.float64)], 
                  cache = False, boundscheck = False)
def linear5_L(J_L, v_ext, gamma, dx):
    '''
    Compute the fifth order approximation to the 
    left convolution integral using a six point global stencil
    and linear weights.
    '''
    # We need gamma*dx here, so we adjust the value of gamma
    gam = gamma*dx 
    
    # Get the total number of elements in v_ext (N = N_ext - 4)
    N_ext = v_ext.size
    
    #----------------------------------------------------------------------------------------------------
    # Compute weights for the quadrature using the precomputed expressions for the left approximation
    #
    # Note: Can precompute these at the beginning of the simulation and load them later for speed
    #----------------------------------------------------------------------------------------------------
    cl_11 = ( 6 - 6*gam + 2*gam**2 - ( 6 - gam**2 )*np.exp(-gam) )/(6*gam**3)
    cl_12 = -( 6 - 8*gam + 3*gam**2 - ( 6 - 2*gam - 2*gam**2 )*np.exp(-gam) )/(2*gam**3)
    cl_13 = ( 6 - 10*gam + 6*gam**2 - ( 6 - 4*gam - gam**2 + 2*gam**3 )*np.exp(-gam) )/(2*gam**3)
    cl_14 = -( 6 - 12*gam + 11*gam**2 - 6*gam**3 - ( 6 - 6*gam + 2*gam**2)*np.exp(-gam) )/(6*gam**3)
    cl_21 = ( 6 - gam**2 - ( 6 + 6*gam + 2*gam**2 )*np.exp(-gam) )/(6*gam**3)
    cl_22 = -( 6 - 2*gam - 2*gam**2 - ( 6 + 4*gam - gam**2 - 2*gam**3 )*np.exp(-gam) )/(2*gam**3)
    cl_23 = ( 6 - 4*gam - gam**2 + 2*gam**3 - ( 6 + 2*gam - 2*gam**2 )*np.exp(-gam) )/(2*gam**3)
    cl_24 = -( 6 - 6*gam + 2*gam**2 - ( 6 - gam**2 )*np.exp(-gam) )/(6*gam**3)
    cl_31 = ( 6 + 6*gam +2*gam**2 - ( 6 + 12*gam + 11*gam**2 + 6*gam**3 )*np.exp(-gam) )/(6*gam**3)
    cl_32 = -( 6 + 4*gam - gam**2 - 2*gam**3 - ( 6 + 10*gam + 6*gam**2 )*np.exp(-gam) )/(2*gam**3 )
    cl_33 = ( 6 + 2*gam - 2*gam**2 - ( 6 + 8*gam + 3*gam**2 )*np.exp(-gam) )/(2*gam**3 )
    cl_34 = -( 6 - gam**2 - ( 6 + 6*gam + 2*gam**2 )*np.exp(-gam) )/(6*gam**3)
        
    #----------------------------------------------------------------------------------------------------
    # Compute the linear WENO weights
    # Note: Can precompute these at the beginning of the simulation and load them later for speed
    #----------------------------------------------------------------------------------------------------
    d1 = ( 60 - 15*gam**2 + 2*gam**4 - ( 60 + 60*gam + 15*gam**2 - 5*gam**3 - 3*gam**4)*np.exp(-gam) )
    d1 = d1/(10*(gam**2)*( 6 - 6*gam + 2*gam**2 - ( 6 - gam**2 )*np.exp(-gam) ) )
    
    d3 = ( 60 - 60*gam + 15*gam**2 + 5*gam**3 - 3*gam**4 - ( 60 - 15*gam**2 + 2*gam**4)*np.exp(-gam) ) 
    d3 = d3/(10*(gam**2)*( 6 - gam**2 - ( 6 + 6*gam + 2*gam**2 )*np.exp(-gam) ) )
    
    d2 = 1 - d1 - d3 
        
    #----------------------------------------------------------------------------------------------------
    # Compute the local integrals J_{i}^{L} on x_{i-1} to x_{i}, i = 1,...,N+1
    #----------------------------------------------------------------------------------------------------

    J_L[0] = 0.0
    
    # Loop through the interior points of the extended array
    # Offset is from the left end-point being excluded
    for i in range(3, N_ext - 2):
        
        # Polynomial interpolants on the smaller stencils
        p1 = cl_11*v_ext[i-3] + cl_12*v_ext[i-2] + cl_13*v_ext[i-1] + cl_14*v_ext[i  ]
        
        p2 = cl_21*v_ext[i-2] + cl_22*v_ext[i-1] + cl_23*v_ext[i  ] + cl_24*v_ext[i+1]
        
        p3 = cl_31*v_ext[i-1] + cl_32*v_ext[i  ] + cl_33*v_ext[i+1] + cl_34*v_ext[i+2]
        
        # Compute the integral using the nonlinear weights and the local polynomials
        J_L[i-2] = d1*p1 + d2*p2 + d3*p3
            
    return None


@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                  nb.float64, nb.float64)], 
                  cache = False, boundscheck = False)
def linear5_R(J_R, v_ext, gamma, dx):
    '''
    Compute the fifth order approximation to the 
    right convolution integral using a six point global stencil
    and linear weights.
    '''
    # We need gamma*dx here, so we adjust the value of gamma
    gam = gamma*dx 
    
    # Get the total number of elements in v_ext (N = N_ext - 4)
    N_ext = v_ext.size
    
    #----------------------------------------------------------------------------------------------------
    # Compute weights for the quadrature using the precomputed expressions for the left approximation
    #
    # Note: Can precompute these at the beginning of the simulation and load them later for speed
    #----------------------------------------------------------------------------------------------------
    cr_34 = ( 6 - 6*gam + 2*gam**2 - ( 6 - gam**2 )*np.exp(-gam) )/(6*gam**3)
    cr_33 = -( 6 - 8*gam + 3*gam**2 - ( 6 - 2*gam - 2*gam**2 )*np.exp(-gam) )/(2*gam**3)
    cr_32 = ( 6 - 10*gam + 6*gam**2 - ( 6 - 4*gam - gam**2 + 2*gam**3 )*np.exp(-gam) )/(2*gam**3)
    cr_31 = -( 6 - 12*gam + 11*gam**2 - 6*gam**3 - ( 6 - 6*gam + 2*gam**2)*np.exp(-gam) )/(6*gam**3)
    cr_24 = ( 6 - gam**2 - ( 6 + 6*gam + 2*gam**2 )*np.exp(-gam) )/(6*gam**3)
    cr_23 = -( 6 - 2*gam - 2*gam**2 - ( 6 + 4*gam - gam**2 - 2*gam**3 )*np.exp(-gam) )/(2*gam**3)
    cr_22 = ( 6 - 4*gam - gam**2 + 2*gam**3 - ( 6 + 2*gam - 2*gam**2 )*np.exp(-gam) )/(2*gam**3)
    cr_21 = -( 6 - 6*gam + 2*gam**2 - ( 6 - gam**2 )*np.exp(-gam) )/(6*gam**3)
    cr_14 = ( 6 + 6*gam +2*gam**2 - ( 6 + 12*gam + 11*gam**2 + 6*gam**3 )*np.exp(-gam) )/(6*gam**3)
    cr_13 = -( 6 + 4*gam - gam**2 - 2*gam**3 - ( 6 + 10*gam + 6*gam**2 )*np.exp(-gam) )/(2*gam**3 )
    cr_12 = ( 6 + 2*gam - 2*gam**2 - ( 6 + 8*gam + 3*gam**2 )*np.exp(-gam) )/(2*gam**3 )
    cr_11 = -( 6 - gam**2 - ( 6 + 6*gam + 2*gam**2 )*np.exp(-gam) )/(6*gam**3)
    
    #----------------------------------------------------------------------------------------------------
    # Compute the linear WENO weights
    #
    # Note: Can precompute these at the beginning of the simulation and load them later for speed
    #----------------------------------------------------------------------------------------------------
    d3 = ( 60 - 15*gam**2 + 2*gam**4 - ( 60 + 60*gam + 15*gam**2 - 5*gam**3 - 3*gam**4)*np.exp(-gam) )
    d3 = d3/(10*(gam**2)*( 6 - 6*gam + 2*gam**2 - ( 6 - gam**2 )*np.exp(-gam) ) )
    
    d1 = ( 60 - 60*gam + 15*gam**2 + 5*gam**3 - 3*gam**4 - ( 60 - 15*gam**2 + 2*gam**4)*np.exp(-gam) ) 
    d1 = d1/(10*(gam**2)*( 6 - gam**2 - ( 6 + 6*gam + 2*gam**2 )*np.exp(-gam) ) )
    
    d2 = 1 - d1 - d3
        
    #----------------------------------------------------------------------------------------------------
    # Compute the local integrals J_{i}^{R} on x_{i} to x_{i+1}, i = 0,...,N
    #----------------------------------------------------------------------------------------------------
    
    # Loop through the interior points
    # Offset is from the right end-point being excluded
    for i in range(2,N_ext - 3):
        
        # Polynomial interpolants on the smaller stencils
        p1 = cr_11*v_ext[i-2] + cr_12*v_ext[i-1] + cr_13*v_ext[i  ] + cr_14*v_ext[i+1]

        p2 = cr_21*v_ext[i-1] + cr_22*v_ext[i  ] + cr_23*v_ext[i+1] + cr_24*v_ext[i+2]

        p3 = cr_31*v_ext[i  ] + cr_32*v_ext[i+1] + cr_33*v_ext[i+2] + cr_34*v_ext[i+3]
        
        # Compute the integral using the nonlinear weights and the local polynomials
        J_R[i-2] = d1*p1 + d2*p2 + d3*p3
    
    J_R[-1] = 0.0

    return None


@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                  nb.float64, nb.float64)], 
                  cache = False, boundscheck = False)
def trapezoid_L(J_L, v, alpha, dx):
    '''
    Compute the first-order approximation to the 
    left convolution integral using a 2 point global stencil
    
    Input:
    =================================================================
    v: A 1-D array for the solution at the time step t_n.
    alpha: Parameter specified by MOLT discretization.
    dx: Grid spacing.
    m_ext: Extension method specified by the boundary conditions
    length: Number of points used in extending the domain
    
    Output:
    =================================================================
    J_L: A 1-D array for the reconstructed left-biased local integrals.
    '''
    # We need nu = alpha*dx and d = exp(-nu)
    nu = alpha*dx 
    d = np.exp(-nu)
    
    # Get the total number of elements in v
    N = v.size
    
    # Compute the quadrature weights
    w_1 = -(d + nu*d - 1)/nu
    w_2 = (d + nu - 1)/nu
    
    #--------------------------------------------------------------------------  
    # Compute the local integrals J_{i}^{L} on x_{i-1} to x_{i}, i = 1,...,N+1
    #--------------------------------------------------------------------------

    J_L[0] = 0.0
    
    # Interior points
    for i in range(1,N):
    
        J_L[i] = w_1*v[i-1] + w_2*v[i]
    
    return None


@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                  nb.float64, nb.float64)], 
                  cache = False, boundscheck = False)
def trapezoid_R(J_R, v, alpha, dx):
    '''
    Compute the first-order approximation to the 
    right convolution integral using a 2 point stencil
    
    Input:
    =================================================================
    v: A 1-D array for the solution at the time step t_n.
    alpha: Parameter specified by MOLT discretization.
    dx: Grid spacing.
    m_ext: Extension method specified by the boundary conditions
    length: Number of points used in extending the domain
    
    Output:
    =================================================================
    J_R: A 1-D array for the reconstructed right-biased local integrals.
    '''
    # We need nu = alpha*dx and d = exp(-nu)
    nu = alpha*dx 
    d = np.exp(-nu)
    
    # Get the total number of elements in v
    N = v.size
    
    # Compute the quadrature weights
    w_1 = (d + nu - 1)/nu
    w_2 = -(d + nu*d - 1)/nu

    #--------------------------------------------------------------------------  
    # Compute the local integrals J_{i}^{R} on x_{i} to x_{i+1}, i = 0,...,N
    #--------------------------------------------------------------------------

    # Interior points
    # Note that J_R[N-1] = 0
    for i in range(0,N-1):
    
        J_R[i] = w_1*v[i] + w_2*v[i+1]
        
    J_R[N-1] = 0.0
            
    return None



