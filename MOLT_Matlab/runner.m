clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

rng(2);

mesh_independent_variable_setup;

grid_refinement = [32];

enable_plots = true;
write_csvs = true;
plot_at = 25;

for g = grid_refinement
    variable_setup;
    results_path = fullfile("results","conserving",tag);
    asym_euler_particle_heating_solver;
end