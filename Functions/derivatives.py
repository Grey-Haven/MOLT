import numba as nb

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64)], 
          cache = True, boundscheck = False)
def compute_ddx_FD(dudx, u, dx):
    """
    Computes an x derivative via finite differences
    """
    
    N_x = u.shape[0]
    N_y = u.shape[1]

    # Left boundary (forward diff)
    for j in range(N_y):
        dudx[0,j] = ( -3*u[0,j] + 4*u[1,j] - u[2,j] )/(2*dx)
        
    # Central derivatives
    for j in range(N_y):
        for i in range(1,N_x-1):
            dudx[i,j] = ( u[i+1,j] - u[i-1,j] )/(2*dx)
            
    # Right boundary (backward diff)
    for j in range(N_y):
        dudx[-1,j] = ( 3*u[-1,j] - 4*u[-2,j] + u[-3,j] )/(2*dx)
    
    return None



@nb.njit([nb.void(nb.float64[:,:], nb.float64[:,:], nb.float64)], 
          cache = True, boundscheck = False)
def compute_ddy_FD(dudy, u, dy):
    """
    Computes an y derivative via finite differences
    """
    
    N_x = u.shape[0]
    N_y = u.shape[1]
    
    # Left boundary (forward diff)
    for i in range(N_x):
        dudy[i,0] = ( -3*u[i,0] + 4*u[i,1] - u[i,2] )/(2*dy)
        
    # Central derivatives
    for i in range(N_x):
        for j in range(1,N_y-1):
            dudy[i,j] = ( u[i,j+1] - u[i,j-1] )/(2*dy)
            
    # Right boundary (backward diff)
    for i in range(N_x):
        dudy[i,-1] = ( 3*u[i,-1] - 4*u[i,-2] + u[i,-3] )/(2*dy)
    
    return None
