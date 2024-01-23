clear;
close all force;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'cmss')
set(0,'DefaultAxesFontName', 'cmss')

grid_refinement = [16,32,64]; % Run FFT BDF BDF for 64x64

debug = true;

enable_plots = false;
write_csvs = false;
plot_at = 50;

gauge_correction_none = "no_mod";
gauge_correction_FFT = "correct_gauge_fft";
gauge_correction_FD6 = "correct_gauge_fd6";

J_rho_update_method_vanilla = "vanilla";
J_rho_update_method_FFT = "FFT";
J_rho_update_method_FD6 = "FD6";

waves_update_method_vanilla = "vanilla";
waves_update_method_FFT = "FFT";
waves_update_method_FD6 = "FD6";

run_type_vanilla_ng = "vanilla_no_gauge_cleaning";
run_type_vanilla_gc = "vanilla_gauge_cleaning";

run_type_FFT_ng = "FFT_no_gauge_cleaning";
run_type_FFT_gc = "FFT_gauge_cleaning";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% THIS IS THE ONLY PARAMETER TO TWEAK
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run_type = run_type_FFT_ng;

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

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_FFT;

    gauge_correction = gauge_correction_none;

elseif run_type == run_type_FFT_gc

    update_method_title = "FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
    update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_FFT;

    gauge_correction = gauge_correction_FFT;

else
    ME = MException('RunException',"No Run of " + run_type + " Type");
    throw(ME);
end

mesh_independent_variable_setup_heating;

g = 16;    
variable_setup_heating;
asym_euler_particle_heating_solver_heating;
t_16 = ts;
var_hist_16 = v_elec_var_hist;
temp_hist_16 = temp_hist;
gauge_error_array_16 = gauge_error_array(:,2);

g = 32;    
variable_setup_heating;
asym_euler_particle_heating_solver_heating;
t_32 = ts;
var_hist_32 = v_elec_var_hist;
temp_hist_32 = temp_hist;
gauge_error_array_32 = gauge_error_array(:,2);

g = 64;    
variable_setup_heating;
asym_euler_particle_heating_solver_heating;
t_64 = ts;
var_hist_64 = v_elec_var_hist;
temp_hist_64 = temp_hist;
gauge_error_array_64 = gauge_error_array(:,2);

g = 128;
variable_setup_heating;
asym_euler_particle_heating_solver_heating;
t_128 = ts;
var_hist_128 = v_elec_var_hist;
temp_hist_128 = temp_hist;
gauge_error_array_128 = gauge_error_array(:,2);

g = 256;
variable_setup_heating;
asym_euler_particle_heating_solver_heating;
t_256 = ts;
var_hist_256 = v_elec_var_hist;
temp_hist_256 = temp_hist;
gauge_error_array_256 = gauge_error_array(:,2);
figure;

x0=200;
y0=200;
width = 1800;
height = 800;
set(gcf,'position',[x0,y0,width,height]);

subplot(1,2,1);
plot(t_16,temp_hist_16);
hold on;
plot(t_32,temp_hist_32);
plot(t_64,temp_hist_64);
plot(t_128,temp_hist_128);
plot(t_256,temp_hist_256);
hold off;
xlabel("t",'FontSize',28);
ylabel("Electron Temperature [K]",'Interpreter','latex','FontSize',28);
legend("16x16", "32x32", "64x64", "128x128", "256x256", 'FontSize',18);
title("Electron Temperature",'FontSize',32);

subplot(1,2,2);
plot(t_16,gauge_error_array_16);
hold on;
plot(t_32,gauge_error_array_32);
plot(t_64,gauge_error_array_64);
plot(t_128,gauge_error_array_128);
plot(t_256,gauge_error_array_256);
hold off;
xlabel("t",'FontSize',28);
ylabel("$||\frac{\partial \phi}{\partial t} + \nabla \cdot \textbf{A}||_2$",'Interpreter','latex','FontSize',28);
legend("16x16", "32x32", "64x64", "128x128", "256x256", 'FontSize',18);
title("Gauge Error",'FontSize',32);

sgtitle("FFT Continuity",'FontSize', 48);

