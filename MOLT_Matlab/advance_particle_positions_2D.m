function [x1_s_new, x2_s_new] = advance_particle_positions_2D(x1_s_new, x2_s_new, ...
                                                              x1_s_old, x2_s_old, ...
                                                              v1_s_mid, v2_s_mid, dt)
    %%%%%%%%%%%
    % Updates particle positions using the mid-point rule.
    %
    % This function should be used with the Boris method. 
    %%%%%%%%%%%
    
    x1_s_new = x1_s_old + dt*v1_s_mid;
    x2_s_new = x2_s_old + dt*v2_s_mid;
end