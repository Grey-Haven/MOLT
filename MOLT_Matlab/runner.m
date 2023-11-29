clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

grid_refinement = [16,32,64];
CFLs = [.01];
particle_count_multipliers = [1];

debug = true;

enable_plots = false;
write_csvs = false;
plot_at = 5;


% update_method_title = "Vanilla";
% update_method_folder = "vanilla";

update_method_title = "Non-Iterative FFT";
update_method_folder = "non_iterative_fft";

% update_method_title = "Iterative FFT";
% update_method_folder = "iterative_fft";

mesh_independent_variable_setup;

for particle_count_multiplier = particle_count_multipliers
    for CFL = CFLs
        for g = grid_refinement
            close all;
            variable_setup;
            results_path = fullfile("results","conserving",tag);
            asym_euler_particle_heating_solver;
        end
    end
end