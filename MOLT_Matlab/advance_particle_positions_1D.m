function x1_s_new = advance_particle_positions_1D(x1_s_old, v1_s_mid, dt)
    %%%%%%%%%%%
    % Updates particle positions using Newton's Second Law.
    %%%%%%%%%%%
    
    x1_s_new = x1_s_old + dt*v1_s_mid;
end