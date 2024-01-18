clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

grid_refinement = [128];
CFLs = [1];
particle_count_multipliers = [1];

debug = true;

enable_plots = true;
write_csvs = false;
plot_at = 5;


update_method_title = "Vanilla";
update_method_folder = "vanilla";

% update_method_title = "FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
% update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

% update_method_title = "FD6 Charge Update, BDF-1 Wave Update, FD6 Derivative";
% update_method_folder = "FD6_charge_BDF_wave_update_FD6_derivative";

modification = "no_mod";
% modification = "FFT_splitting_err";
% modification = "correct_gauge";
% modification = "correct_gauge_fft";

mesh_independent_variable_setup_weibel;

for particle_count_multiplier = particle_count_multipliers
    for CFL = CFLs
        for g = grid_refinement
            close all;
            variable_setup_weibel_quiet;
            asym_euler_particle_heating_solver_weibel;
        end
    end
end