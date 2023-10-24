function [] = advance_particle_positions_2D(x1_s_new, x2_s_new, ...
                                            x1_s_old, x2_s_old, ...
                                            v1_s_mid, v2_s_mid, dt)
    %%%%%%%%%%%
    % Updates particle positions using the mid-point rule.
    %
    % This function should be used with the Boris method. 
    %%%%%%%%%%%
    
    % Number of particles of a species s
    N_s = length(x1_s_old);
    
    for i = 1:N_s        
        x1_s_new(i) = x1_s_old(i) + dt*v1_s_mid(i);
        x2_s_new(i) = x2_s_old(i) + dt*v2_s_mid(i);
    end
end