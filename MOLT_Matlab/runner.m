clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

grid_refinement = [64]; % Run FFT BDF BDF for 64x64
CFLs = [1];
particle_count_multipliers = [10];

debug = true;

enable_plots = false;
write_csvs = false;
plot_at = 500;


% update_method_title = "Vanilla";
% update_method_folder = "vanilla";

% update_method_title = "FD6 Charge Update, BDF-1 Wave Update, FD6 Derivative";
% update_method_folder = "FD6_charge_BDF_wave_update_FD6_derivative";

update_method_title = "FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";
    
modification = "no_mod";
% modification = "correct_gauge_fft";

mesh_independent_variable_setup;

for particle_count_multiplier = particle_count_multipliers
    for CFL = CFLs
        for g = grid_refinement
            close all;
            variable_setup;
            asym_euler_particle_heating_solver;
        end
    end
end