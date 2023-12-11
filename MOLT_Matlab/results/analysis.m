clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

mesh_independent_variable_setup;

grid_refinement = [16,32,64];
CFLs = [1];
particle_count_multipliers = [10];

debug = false;

% update_method_title = "FD6 Charge Update, FFT Wave Update, FFT Derivative";
% update_method_folder = "FD6_charge_FFT_wave_update_FFT_derivative";

% update_method_title = "FD6 Charge Update, FFT Derivative";
% update_method_folder = "FD6_charge_BDF_wave_update_FFT_derivative";

% update_method_title = "      FD6 Charge Update, FD6 Derivative";
% update_method_folder = "FD6_charge_BDF_wave_update_FD6_derivative";

% update_method_title = "FD6 Charge Update, FFT Derivative";
% update_method_folder = "FD6_charge_BDF_wave_update_FFT_derivative";

% update_method_title = "FFT Charge Update, FD6 Derivative";
% update_method_folder = "FFT_charge_BDF_wave_update_FD6_derivative";

% update_method_title = "      FFT Charge Update, BDF-1 Derivative";
% update_method_folder = "FFT_charge_BDF_wave_update_BDF_derivative";

% update_method_title = "      FFT Charge Update, FFT Derivative";
% update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

update_method_title = "      Naive Charge Map, BDF-1 Derivative";
update_method_folder = "vanilla";

% update_method_title = "Non-Iterative FFT";
% update_method_folder = "non_iterative_fft";

modification = "no_mod";
% modification = "correct_gauge";

filePath = matlab.desktop.editor.getActiveFilename;
projectRoot = fileparts(filePath);

labels = string(length(grid_refinement) * length(CFLs) * length(particle_count_multipliers));

l = 1;

figure;
x0=200;
y0=100;
width = 1200;
height = 1200;
set(gcf,'position',[x0,y0,width,height]);

titleFontSize = 32;
tickFontSize = 28;
subTickFontSize = 16;
legendFontSize = 20;
lineWidth = 1;

for particle_count_multiplier = particle_count_multipliers
    for CFL = CFLs
        for g = grid_refinement

            variable_setup;
            ts = 0:dt:T_final-dt;
            
            tag = g + "x" + g;
            path = projectRoot + "/conserving/p_mult_" + particle_count_multiplier + ...
                   "/CFL_" + CFL + "/" + modification + "/" + update_method_folder + "/" + tag + "/";
            csvPath = path + "csv_files/" + "gauge_error.csv";
            gauge_error = readmatrix(csvPath);

            ts = gauge_error(:,1);
            l2_err = gauge_err(:,2);
            inf_err = gauge_err(:,3);
            
            plot(ts,l2_err,'LineWidth',lineWidth);
            plot(ts,inf_err,'LineWidth',lineWidth);
            hold on;
            
            labels(l) = tag;
            
            l = l+1;
        end
    end
end
ax = gca;
ax.FontSize = tickFontSize;

legend(labels, 'FontSize', legendFontSize);
title({"Gauge Error", update_method_title}, 'FontSize', titleFontSize);
xlabel("t", 'FontSize', tickFontSize);
ylabel("L_2 Error", 'FontSize', tickFontSize);
ylim([0,2.5e-13]);
% 
% 
% axes('position',[.45 .175 .4 .25])
% box on % put box around new pair of axes
% 
% for particle_count_multiplier = particle_count_multipliers
%     for CFL = CFLs
%         for g = grid_refinement
% 
%             variable_setup;
%             ts = 0:dt:T_final-dt;
% 
%             tag = g + "x" + g;
%             path = projectRoot + "/conserving/p_mult_" + particle_count_multiplier + ...
%                    "/CFL_" + CFL + "/" + modification + "/" + update_method_folder + "/" + tag + "/";
%             csvPath = path + "csv_files/" + "gauge_error.csv";
%             gauge_error = readmatrix(csvPath);
% 
% 
%             indexOfInterest = (gauge_error(:,1) > .25); % range of t near perturbation
%             plot(gauge_error(indexOfInterest,1),gauge_error(indexOfInterest,2)) % plot on new axes
%             hold on;
%         end
%     end
% end

% ax = gca;
% ax.FontSize = subTickFontSize;

saveas(gcf, projectRoot + "/conserving/p_mult_" + particle_count_multiplier + "/CFL_" + CFL + "/" + modification + "/" + update_method_folder + "/" + update_method_folder + "_fig.jpg");