import numpy as np

def remove_particles(x1_old, x2_old,
                     x1_new, x2_new,
                     v1_old, v2_old,
                     v1_new, v2_new,
                     P1_old, P2_old,
                     P1_new, P2_new,
                     v1_nm1, v2_nm1,
                     ax, bx, ay, by):
    """
    "Removes" particles from the simulation whose (x1,x2) coordinates
    are outside of the bounding box [ax, bx] x [ay, by].
    
    We assume also that the input arrays are valid slices that correspond
    to the array indices where particles are being tracked.
    
    We need all the time levels for the particle data to ensure that
    the time history will not be modified unintentionally.
    """
    
    N_part = x1_old.size
    
    # Create temoraries to store the sorted output
    # which allows us to pass by reference. Otherwise,
    # the output will not be correctly returned
    x1_tmp = np.zeros_like(x1_new) 
    x2_tmp = np.zeros_like(x2_new)
    
    v1_tmp = np.zeros_like(v1_new) 
    v2_tmp = np.zeros_like(v2_new)
    
    P1_tmp = np.zeros_like(P1_new) 
    P2_tmp = np.zeros_like(P2_new)
    
    # Check where the new particle coordinates are outside the domain
    out_x1_left = np.argwhere(x1_new < ax)
    out_x1_rite = np.argwhere(x1_new > bx)

    out_x2_left = np.argwhere(x2_new < ay)
    out_x2_rite = np.argwhere(x2_new > by)
    
    # A particle is outside the domain if at least one of its coordinates is outside
    out_idx = np.concatenate((out_x1_left, out_x1_rite, out_x2_left, out_x2_rite))
    out_idx = np.unique(out_idx)
    N_part_out = out_idx.size
    
    # Sort this boolean array by it's indices so that particles outside are
    # placed at the end of the array
    is_outside = np.zeros([x1_new.size], dtype=bool)
    is_outside[out_idx] = True
    
    sort_idx = np.argsort(is_outside) # Elements = True will be at the end
    
    # Sort the position and velocity arrays using this index mapping
    # then read back into the input arrays
    
    # New time level
    x1_tmp[:] = x1_new[:] 
    x2_tmp[:] = x2_new[:]
    v1_tmp[:] = v1_new[:]
    v2_tmp[:] = v2_new[:]
    P1_tmp[:] = P1_new[:]
    P2_tmp[:] = P2_new[:]
    
    x1_new[:] = x1_tmp[sort_idx] 
    x2_new[:] = x2_tmp[sort_idx]
    v1_new[:] = v1_tmp[sort_idx]
    v2_new[:] = v2_tmp[sort_idx]
    P1_new[:] = P1_tmp[sort_idx]
    P2_new[:] = P2_tmp[sort_idx]
    
    # Old time level
    x1_tmp[:] = x1_old[:] 
    x2_tmp[:] = x2_old[:]
    v1_tmp[:] = v1_old[:]
    v2_tmp[:] = v2_old[:]
    P1_tmp[:] = P1_old[:]
    P2_tmp[:] = P2_old[:]
    
    x1_old[:] = x1_tmp[sort_idx] 
    x2_old[:] = x2_tmp[sort_idx]
    v1_old[:] = v1_tmp[sort_idx]
    v2_old[:] = v2_tmp[sort_idx]
    P1_old[:] = P1_tmp[sort_idx]
    P2_old[:] = P2_tmp[sort_idx]
    
    # Need to also do this for the additional time history v^{n-1}
    v1_tmp[:] = v1_nm1[:]
    v2_tmp[:] = v2_nm1[:]
    
    v1_nm1[:] = v1_tmp[sort_idx] 
    v2_nm1[:] = v2_tmp[sort_idx]
    
    # Update the number of particles
    N_part -= N_part_out
  
    return N_part
