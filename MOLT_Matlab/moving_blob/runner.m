clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));
addpath(genpath([fileparts(pwd), '/wave_solvers']));

set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'cmss')
set(0,'DefaultAxesFontName', 'cmss')

grid_refinement = [16,32,64]; % Run FFT BDF BDF for 64x64
CFLs = [1];
particle_count_multipliers = [10];

debug = true;

enable_plots = true;
write_csvs = false;
plot_at = 50;
gauge_correction_none = "no_mod";
gauge_correction_FFT = "correct_gauge_fft";
gauge_correction_FD6 = "correct_gauge_fd6";

J_rho_update_method_vanilla = "vanilla";
J_rho_update_method_FFT = "FFT";
J_rho_update_method_FD6 = "FD6";

J_rho_update_method_BDF1_FFT = "BDF1-FFT";
J_rho_update_method_BDF2_FFT = "BDF2-FFT";
J_rho_update_method_BDF3_FFT = "BDF3-FFT";
J_rho_update_method_BDF4_FFT = "BDF4-FFT";
J_rho_update_method_CDF1_FFT = "CDF1-FFT";

waves_update_method_vanilla = "vanilla";
waves_update_method_FFT = "FFT";
waves_update_method_FD6 = "FD6";
waves_update_method_poisson_phi = "poisson_phi";
waves_update_method_pure_FFT = "pure_FFT";

J_rho_update_method_CDF1_FFT = "CDF1-FFT";

run_type_vanilla_ng = "vanilla_no_gauge_cleaning";
run_type_vanilla_gc = "vanilla_gauge_cleaning";

run_type_FFT_ng = "FFT_no_gauge_cleaning";
run_type_FFT_gc = "FFT_gauge_cleaning";

run_type_FD6_ng = "FD6_no_gauge_cleaning";
run_type_FD6_gc = "FD6_gauge_cleaning";

run_type_pure_FFT_ng = "pure_FFT_no_gauge_cleaning";
run_type_pure_FFT_gc = "pure_FFT_gauge_cleaning";

run_type_poisson_ng = "FFT_A_poisson_phi_no_gauge_cleaning";

J_rho_BDF_FFT_Family = [J_rho_update_method_BDF1_FFT, J_rho_update_method_BDF2_FFT, J_rho_update_method_BDF3_FFT, J_rho_update_method_BDF4_FFT, J_rho_update_method_CDF1_FFT];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% THIS IS THE ONLY PARAMETER TO TWEAK
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run_type = run_type_FFT_ng;
run_type = run_type_FD6_ng;
% run_type = run_type_poisson_ng;
% run_type = run_type_pure_FFT_ng;
% run_type = run_type_vanilla_ng;

if run_type == run_type_vanilla_ng

    update_method_title = "Vanilla";
    update_method_folder = "vanilla";

    J_rho_update_method = J_rho_update_method_vanilla;
    waves_update_method = waves_update_method_vanilla;

    gauge_correction = gauge_correction_none;

elseif run_type == run_type_vanilla_gc

    update_method_title = "Vanilla";
    update_method_folder = "vanilla";

    J_rho_update_method = J_rho_update_method_vanilla;
    waves_update_method = waves_update_method_vanilla;

    gauge_correction = gauge_correction_FFT;

elseif run_type == run_type_FFT_ng

    update_method_title = "FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
    update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

    J_rho_update_method = J_rho_update_method_BDF1_FFT;
    waves_update_method = waves_update_method_FFT;

    gauge_correction = gauge_correction_none;

elseif run_type == run_type_FFT_gc

    update_method_title = "FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
    update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_FFT;

    gauge_correction = gauge_correction_FFT;

elseif run_type == run_type_FD6_ng

    update_method_title = "FD6 Charge Update, BDF-1 Wave Update, FD6 Derivative";
    update_method_folder = "FD6_charge_BDF_wave_update_FD6_derivative";

    J_rho_update_method = J_rho_update_method_FD6;
    waves_update_method = waves_update_method_FD6;

    gauge_correction = gauge_correction_none;

elseif run_type == run_type_FD6_gc

    update_method_title = "FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
    update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_FFT;

    gauge_correction = gauge_correction_FFT;

elseif run_type == run_type_poisson_ng

    update_method_title = "FFT Charge Update, BDF-1 A Update, Poisson Phi Update";
    update_method_folder = "FFT_charge_BDF1_A_poisson_phi";

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_poisson_phi;

    gauge_correction = gauge_correction_none;
elseif run_type == run_type_pure_FFT_ng

    update_method_title = "Second Order FFT Charge Update, FFT Potentials, FFT Derivatives";
    update_method_folder = "FFT_charge_pure_FFT";

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_pure_FFT;

    gauge_correction = gauge_correction_none;
elseif run_type == run_type_pure_FFT_gc

    update_method_title = "Second Order FFT Charge Update, FFT Potentials, FFT Derivatives";
    update_method_folder = "FFT_charge_BDF1_A_poisson_phi";

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_pure_fft;

    gauge_correction = gauge_correction_FFT;
else
    ME = MException('RunException',"No Run of " + run_type + " Type");
    throw(ME);
end

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