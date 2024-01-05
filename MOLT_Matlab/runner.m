clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

grid_refinement = [16,32,64]; % Run FFT BDF BDF for 64x64
CFLs = [1];
particle_count_multipliers = [10];

debug = true;

enable_plots = true;
write_csvs = true;
plot_at = 500;


update_method_title = "Vanilla";
update_method_folder = "vanilla";

% update_method_title = "FFT Charge Update, BDF-1 Wave Update, FD6 Derivative";
% update_method_folder = "FFT_charge_BDF_wave_update_FD6_derivative";

% update_method_title = "FFT Charge Update, BDF-1 Wave Update, BDF-1 Derivative";
% update_method_folder = "FFT_charge_BDF_wave_update_BDF_derivative";

% update_method_title = "FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
% update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

% update_method_title = "FD6 Charge Update, FFT Wave Update, FFT Derivative";
% update_method_folder = "FD6_charge_FFT_wave_update_FFT_derivative";

% update_method_title = "FD6 Charge Update, BDF-1 Wave Update, FFT Derivative";
% update_method_folder = "FD6_charge_BDF_wave_update_FFT_derivative";

% update_method_title = "FD6 Charge Update, BDF-1 Wave Update, FD6 Derivative, Gauge Corrected";
% update_method_folder = "FD6_charge_BDF_wave_update_FD6_derivative";

% update_method_title = "FD6 Charge Update, BDF-1 Wave Update, FFT Derivative";
% update_method_folder = "FD6_charge_BDF_wave_update_FFT_derivative";

% update_method_title = "Vanilla Particle Update, FFT Wave Update";
% update_method_folder = "vanilla_particle_fft_wave_update";

% update_method_title = "Non-Iterative FFT";
% update_method_folder = "non_iterative_fft";

% update_method_title = "Iterative FFT";
% update_method_folder = "iterative_fft";
    
% modification = "no_mod";
% modification = "FFT_splitting_err";
modification = "correct_gauge";

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