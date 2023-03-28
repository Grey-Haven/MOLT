import numpy as np
import numba as nb

@nb.njit([nb.void(nb.float64[:], nb.float64[:], 
                 nb.float64, nb.float64, nb.float64)], 
                 cache= False, boundscheck = False)
def apply_A_and_B(I, x, alpha, A, B):
    
    N = x.size
    
    for i in range(N):
        
        I[i] += A*np.exp(-alpha*( x[i ] - x[0] ))
        I[i] += B*np.exp(-alpha*( x[-1] - x[i] ))
    
    return None

@nb.njit([nb.float64[:](nb.float64)], cache= False, boundscheck = False)
def get_explicit_BDF2_outflow_wts(beta):
    """
    Computes the second-order explicit outflow integration weights for the BDF-2 scheme.
    
    The first entry corresponds to the most recent time point.
    """
    
    gamma = np.zeros((3)) # Uses { u^{n-2}, u^{n-1}, u^{n}} } to construct the interpolant
    
    # Formulas below are computed through Lagrange interpolation
    epb = np.exp( beta)
    emb = np.exp(-beta)
    beta2 = beta**2
    
    gamma[0] = -(5*beta + 2*emb + beta2*emb - 5*beta2 - 3*beta*emb - 2)/(4*beta2)
    gamma[1] = -(beta2*emb - 2*emb - 4*beta + 2*beta2 + 2*beta*emb + 2)/(2*beta2)
    gamma[2] = (emb*(- 1 + beta)*(beta - 2*epb + beta*epb + 2))/(4*beta2)
    
    return gamma

@nb.njit([nb.float64[:](nb.float64)], cache= False, boundscheck = False)
def get_implicit_BDF2_outflow_wts(beta):
    """
    Computes the second-order implicit outflow integration weights for the BDF-2 scheme.
    
    The first entry corresponds to the most recent time point.
    """
    
    gamma = np.zeros((4)) # Uses { u^{n-2}, ..., u^{n+1}} } to construct the interpolant
    
    # Formulas below are computed through Lagrange interpolation
    emb = np.exp(-beta)
    beta2 = beta**2
    beta3 = beta2*beta
    
    gamma[0] = -(beta2*emb - 6*emb - 12*beta - 3*beta3*emb + 8*beta2 + 6*beta*emb + 6)/(12*beta3)
    gamma[1] = (4*beta2*emb - 6*emb - 10*beta - 4*beta3*emb + 3*beta2 + 5*beta3 + 4*beta*emb + 6)/(4*beta3)
    gamma[2] = -(5*beta2*emb - 6*emb - 8*beta - beta3*emb + 4*beta3 + 2*beta*emb + 6)/(4*beta3)
    gamma[3] = -(6*beta + 6*emb - 4*beta2*emb + beta2 - 3*beta3 - 6)/(12*beta3)

    return gamma

@nb.njit([nb.float64[:](nb.float64)], cache= False, boundscheck = False)
def get_explicit_central2_outflow_wts(beta):
    """
    Computes the second-order explicit outflow integration weights for the central-2 scheme.
    
    The first entry corresponds to the most recent time point.
    """
    
    # Method uses { u^{n-2}, u^{n-1}, u^{n}} } to construct the interpolant
    gamma = np.zeros((3)) # Modified weights for the central scheme
    
    # Commonly used variables
    beta2 = beta**2
    emb = np.exp(-beta)
    
    # Formulas below are computed through Lagrange interpolation
    gamma[0] = (2*emb - beta + 2*beta2 + 3*beta*emb - 2)/(4*beta2)
    gamma[1] = -(2*emb - 2*beta + 3*beta2*emb + 4*beta*emb - 2)/(6*beta2)
    gamma[2] = -(beta + 2*emb + beta*emb - 2)/(12*beta2)

    return gamma

@nb.njit([nb.float64[:](nb.float64)], cache= False, boundscheck = False)
def get_implicit_central2_outflow_wts(beta):
    """
    Computes the second-order implicit outflow integration weights for the central-2 scheme.
    
    The first entry corresponds to the most recent time point.
    """
    
    # Method uses { u^{n-1}, u^{n}, u^{n+1}} } to construct the interpolant
    Gamma = np.zeros((3)) # Modified weights for the central scheme
    
    # Commonly used variables
    beta2 = beta**2
    emb = np.exp(-beta)
    
    # Formulas below are computed through Lagrange interpolation
    gamma_0 = -(beta + 2*emb + beta*emb - 2)/(4*beta2)
    gamma_1 = (2*emb + beta2 + 2*beta*emb - 2)/(2*beta2)
    gamma_2 = -(2*emb - beta + 2*beta2*emb + 3*beta*emb - 2)/(4*beta2)

    Gamma[0] = beta2*gamma_0
    Gamma[1] = gamma_1 - gamma_0*(beta2 - 2)
    Gamma[2] = gamma_2 - gamma_0

    return Gamma

@nb.njit([nb.float64[:](nb.float64)], cache= False, boundscheck = False)
def get_explicit_central4_outflow_wts(beta):
    """
    Computes the fourth-order explicit outflow integration weights for the central-4 scheme.
    
    The first entry corresponds to the most recent time point.
    """
    
    gamma = np.zeros((5)) # Uses { u^{n-4}, ..., u^{n}} } to construct the interpolant
    
    # Commonly used variables
    beta2 = beta**2
    beta3 = beta**3
    beta4 = beta**4
    
    emb = np.exp(-beta)
    epb = np.exp( beta)
    
    # Formulas below are computed through Lagrange interpolation
    gamma[0] = (3*beta3*emb - 12*emb - 11*beta2*emb - 30*beta + 35*beta2 
                - 25*beta3 + 12*beta4 + 18*beta*emb + 12)/(24*beta4)
    
    gamma[1] = (emb*(54*beta*epb - 24*epb - 30*beta + 10*beta2 + 5*beta3 
                - 6*beta4 - 52*beta2*epb + 24*beta3*epb + 24))/(12*beta4)
    
    gamma[2] = -(24*beta + 12*emb + beta2*emb + 3*beta3*emb - 19*beta2 
                 + 6*beta3 - 12*beta*emb - 12)/(4*beta4)
 
    gamma[3] = (42*beta + 24*emb - 2*beta2*emb + 3*beta3*emb - 28*beta2 
                + 8*beta3 - 18*beta*emb - 24)/(12*beta4)
    
    gamma[4] = -(18*beta + 12*emb - beta2*emb + beta3*emb - 11*beta2 
                 + 3*beta3 - 6*beta*emb - 12)/(24*beta4)
    
    return gamma


@nb.njit([nb.types.UniTuple(nb.float64,2)(nb.float64, nb.float64, 
                                          nb.float64[:], nb.float64[:],
                                          nb.float64, nb.float64, nb.float64[:],
                                          nb.float64, nb.float64)], 
                                          cache= False, boundscheck = False)
def apply_level1_out(I_a, I_b, v_bdry_a, v_bdry_b, A_nm1, B_nm1, Gamma, mu, beta):
    """
    Level 1 outflow BCs for the central-2 scheme are computed with the implicit, second-order form 
    for the integration weights.
    
    Note: I is the convolution integral (along a line) and v_bdry_a and v_bdry_b is the time history of the 
    solution at the boundary points along a given direction.
    """
    
    assert Gamma.size == 3, "The level 1 operation should have 3 integration weights.\n"
    assert v_bdry_a.size == 2 and v_bdry_b.size == 2, "The level 1 operation requires last 2 time levels.\n"
    
    # Common constant
    emb = np.exp(-beta)
    
    # Boundary history of the solution
    v_nm1_a = v_bdry_a[0]
    v_n_a = v_bdry_a[1]
    
    v_nm1_b = v_bdry_b[0]
    v_n_b = v_bdry_b[1]
    
    # Compute the right-hand side of the linear system for the boundary conditions
    w_a_out = emb*A_nm1 + Gamma[0]*I_a + Gamma[1]*v_n_a + Gamma[2]*v_nm1_a
    w_b_out = emb*B_nm1 + Gamma[0]*I_b + Gamma[1]*v_n_b + Gamma[2]*v_nm1_b
    
    # Compute the boundary coefficients A and B for the scheme obtained by solving this system
    An = ( (1 - Gamma[0])*w_a_out + mu*Gamma[0]*w_b_out )/( (1 - Gamma[0])**2 - (mu*Gamma[0])**2 )
    Bn = ( (1 - Gamma[0])*w_b_out + mu*Gamma[0]*w_a_out )/( (1 - Gamma[0])**2 - (mu*Gamma[0])**2 )
    
    return An, Bn


@nb.njit([nb.types.UniTuple(nb.float64,2)(nb.float64[:], nb.float64[:],
                                          nb.float64, nb.float64, 
                                          nb.float64[:], nb.float64)], 
                                          cache= False, boundscheck = False)
def apply_level2_out(v_bdry_a, v_bdry_b, A_nm1, B_nm1, gamma, beta):
    """
    Level 2 outflow BCs for the central-4 scheme are computed with the explicit, fourth-order form 
    for the integration weights.
    
    Note: v_bdry_a and v_bdry_b is the time history of the 
    solution at the boundary points along a given direction.
    """
    
    assert gamma.size == 5, "The level 2 operation should have 5 integration weights.\n"
    assert v_bdry_a.size == 5 and v_bdry_b.size == 5, "The level 2 operation requires 5 time levels.\n"
    
    # Common constant
    emb = np.exp(-beta)
    
    # Compute the boundary coefficients A and B for the scheme using an explicit formula with the recursion
    # Note that due to our convention, the gamma array should be reversed
    An = emb*A_nm1 + np.dot(gamma[::-1], v_bdry_a)
    Bn = emb*B_nm1 + np.dot(gamma[::-1], v_bdry_b)
    
    return An, Bn