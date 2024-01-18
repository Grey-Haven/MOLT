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

% update_method_title = "Vanilla";
% update_method_folder = "vanilla";

% update_method_title = "FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
% update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

update_method_title = "FD6 Charge Update, BDF-1 Wave Update, FD6 Derivative";
update_method_folder = "FD6_charge_BDF_wave_update_FD6_derivative";

modification = "no_mod";
% modification = "FFT_splitting_err";
% modification = "correct_gauge";
% modification = "correct_gauge_fft";

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

figure;

x0=200;
y0=200;
width = 1800;
height = 800;
set(gcf,'position',[x0,y0,width,height]);

subplot(1,2,1);
plot(t_16,temp_hist_16);
% hold on;
% plot(t_32,temp_hist_32);
% plot(t_64,temp_hist_64);
% hold off;
xlabel("t",'FontSize',28);
ylabel("Electron Temperature [K]",'Interpreter','latex','FontSize',28);
% legend("16x16", "32x32", "64x64",'FontSize',18);
title("Electron Temperature",'FontSize',32);

subplot(1,2,2);
plot(t_16,gauge_error_array_16);
% hold on;
% plot(t_32,gauge_error_array_32);
% plot(t_64,gauge_error_array_64);
% hold off;
xlabel("t",'FontSize',28);
ylabel("$||\frac{\partial \phi}{\partial t} + \nabla \cdot \textbf{A}||_2$",'Interpreter','latex','FontSize',28);
% legend("16x16", "32x32", "64x64",'FontSize',18);
title("Gauge Error",'FontSize',32);

sgtitle("16x16 Vanilla / FD6 Gauge Correction",'FontSize', 48)

