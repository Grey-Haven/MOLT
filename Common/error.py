import numpy as np

# L_1 error
def get_L_1_error(U_numerical, U_exact, delta):
    
    return delta*np.sum( np.abs(U_numerical - U_exact) )

# L_2 error
def get_L_2_error(U_numerical, U_exact, delta):
    
    return np.sqrt( delta*np.sum( (U_numerical - U_exact)**2 ) )

# L_infinity error
def get_L_infinity_error(U_numerical, U_exact):
    
    return np.max( np.abs( U_numerical - U_exact ) )