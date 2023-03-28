import numba as nb

from particle_funcs import scatter_2D

@nb.njit([nb.void(nb.float64[:,:,:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64[:,:], nb.float64)], 
                  cache=False, boundscheck=False)
def map_J_to_mesh_2D2V(J_mesh, x, y, dx, dy,
                       x1, x2, v1, v2,
                       q_s, cell_volumes, w_s):
    """
    Computes the current density for the field solvers using velocity information
    in the 2D-2V setting.
    
    This mapping is to be used for the expanding beam problem.
    """
    
    assert(x1.size == x2.size)
    assert(x1.size == v1.size)
    assert(x1.size == v2.size)

    # Number of simulation particles
    N_part = x1.size
    
    weight = w_s*q_s

    # Scatter current to the mesh
    for i in range(N_part):
        
        weight1 = weight*v1[i]
        weight2 = weight*v2[i]

        
        scatter_2D(J_mesh[0,:,:], x1[i], x2[i], x, y, dx, dy, weight1) # J_x
        scatter_2D(J_mesh[1,:,:], x1[i], x2[i], x, y, dx, dy, weight2) # J_y
       
    # End of particle loop
    
    # Divide by the cell volumes to compute the number density
    # Should be careful for the multi-species case. If this function
    # is called for several species, the division occurs multiple times.
    # For this, we can either: Do the division outside of this function or
    # create a rho for each particle species and apply this function (unchanged).
    J_mesh[0,:,:] /= cell_volumes[:,:]
    J_mesh[1,:,:] /= cell_volumes[:,:]
    
    # BCs are not periodic
    
    return None

@nb.njit([nb.void(nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                  nb.float64[:], nb.float64[:],
                  nb.float64, nb.float64[:,:], nb.float64)], 
                  cache=False, boundscheck=False)
def map_rho_to_mesh_2D(rho_mesh, x, y, dx, dy,
                       x1, x2,
                       q_s, cell_volumes, w_s):
    """
    Computes the charge density on the mesh using 
    the standard single level spline maps.
    
    Assumes a single species is present
    """
    
    # Number of simulation particles
    N_part = x1.size
    
    weight = w_s*q_s
        
    # Scatter particle charge data to the mesh
    for i in range(N_part):
        
        scatter_2D(rho_mesh[:,:], x1[i], x2[i], x, y, dx, dy, weight)
        
    # End of particle loop
    
    # Divide by the cell volumes to compute the number density
    # Should be careful for the multi-species case. If this function
    # is called for several species, the division occurs multiple times.
    # For this, we can either: Do the division outside of this function or
    # create a rho for each particle species and apply this function (unchanged).
    rho_mesh[:,:] /= cell_volumes[:,:]
    
    # BCs are not periodic
    
    return None