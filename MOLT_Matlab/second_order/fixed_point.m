% The situation:
%
% We have phi, rho, and x at t^{0}.
% We have A, J, and v at t^{1/2}.
%
% We need a greater time history, namely reaching back to t^{-1} for the 
% integer timesteps and t^{-3/2} for the half steps.
% Thus we step forward a fixed amount to t^{20}, and then 
% step backwards to t^{-3/2}. This provides us with a pseudo time history.

N_prep_steps = 20;

t_n = 0;
t_half = dt/2;

for i = 1:N_prep_steps
    full_step_forward;
    create_plots(x, y, psi, A1, A2, rho_mesh(:,:,end), ...
                 gauge_residual, gauss_residual, ...
                 x1_elec_new, x2_elec_new, t_n, ...
                 update_method_title, tag, vidObj);
    t_n = t_n + dt;
    t_half = t_half + dt;

    create_plots(x, y, psi, A1, A2, rho_mesh(:,:,end), ...
                 gauge_residual, gauss_residual, ...
                 x1_elec_new, x2_elec_new, t_n, ...
                 update_method_title, tag, vidObj);
end

for i = N_prep_steps:-1:-1
    full_step_reverse;
    create_plots(x, y, psi, A1, A2, rho_mesh(:,:,beg), ...
                 gauge_residual, gauss_residual, ...
                 x1_elec_new, x2_elec_new, t_n, ...
                 update_method_title, tag, vidObj);
    t_n = t_n - dt;
    t_half = t_half - dt;
end

disp('foo');