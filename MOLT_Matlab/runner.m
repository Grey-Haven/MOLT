clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
% addpath(genpath([fileparts(pwd), '/common']));
% addpath(genpath([fileparts(pwd), '/basic_boris']));

rng(2);

mesh_independent_variable_setup;

grid_refinement = [32];

enable_plots = true;
plot_at = 25;

for g = grid_refinement
    variable_setup;
    results_path = fullfile("results","conserving",tag);

    asym_euler_particle_heating_solver(x1_ions, x2_ions, ...
                                       P1_ions, P2_ions, ...
                                       v1_ions, v2_ions, ...
                                       x1_elec, x2_elec, ...
                                       P1_elec, P2_elec, ...
                                       v1_elec, v2_elec, ...
                                       x, y, dx, dy, kappa, T_final, N_steps, ...
                                       q_ions, q_elec, ...
                                       r_ions, r_elec, ...
                                       w_ions, w_elec, ...
                                       sigma_1, sigma_2, ...
                                       results_path, ...
                                       enable_plots, ...
                                       plot_at);
end