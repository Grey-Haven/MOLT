clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/../../utility_functions']));
addpath(genpath([fileparts(pwd), '/../../source_updaters']));
addpath(genpath([fileparts(pwd), '/../wave_solvers']));

set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'cmss')
set(0,'DefaultAxesFontName', 'cmss')

grid_refinement = [16]; % Run FFT BDF BDF for 64x64
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
J_rho_update_method_DIRK2 = "DIRK2";
J_rho_update_method_FD2 = "FD2";
J_rho_update_method_FD4 = "FD4";
J_rho_update_method_FD6 = "FD6";

waves_update_method_vanilla = "vanilla";
waves_update_method_FFT = "FFT";
waves_update_method_DIRK2 = "DIRK2";
waves_update_method_FD2 = "FD2";
waves_update_method_FD4 = "FD4";
waves_update_method_FD6 = "FD6";
waves_update_method_poisson_phi = "poisson_phi";

waves_update_method_pure_FFT = "pure_fft";

run_type_vanilla_ng = "vanilla_no_gauge_cleaning";
run_type_vanilla_gc = "vanilla_gauge_cleaning";

run_type_FFT_ng = "FFT_no_gauge_cleaning";
run_type_FFT_gc = "FFT_gauge_cleaning";

run_type_pure_FFT_ng = "pure_FFT_no_gauge_cleaning";
run_type_pure_FFT_gc = "pure_FFT_gauge_cleaning";

run_type_poisson_ng = "FFT_A_poisson_phi_no_gauge_cleaning";

run_type_DIRK2_ng = "DIRK_FFT_deriv_no_gauge_cleaning";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% THIS IS THE ONLY PARAMETER TO TWEAK
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run_type = run_type_vanilla_ng;
% run_type = run_type_pure_FFT_ng;
% run_type = run_type_FFT_ng;
% run_type = run_type_poisson_ng;
run_type = run_type_DIRK2_ng;

if run_type == run_type_vanilla_ng

    update_method_title = "Second Order Vanilla";
    update_method_folder = "vanilla";

    J_rho_update_method = J_rho_update_method_vanilla;
    waves_update_method = waves_update_method_vanilla;

    gauge_correction = gauge_correction_none;

elseif run_type == run_type_vanilla_gc

    update_method_title = "Second Order Vanilla";
    update_method_folder = "vanilla";

    J_rho_update_method = J_rho_update_method_vanilla;
    waves_update_method = waves_update_method_vanilla;

    gauge_correction = gauge_correction_FFT;

elseif run_type == run_type_FFT_ng

    update_method_title = "Second Order FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
    update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_FFT;

    gauge_correction = gauge_correction_none;

elseif run_type == run_type_FFT_gc

    update_method_title = "Second Order FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
    update_method_folder = "FFT_charge_BDF_wave_update_FFT_derivative";

    J_rho_update_method = J_rho_update_method_FFT;
    waves_update_method = waves_update_method_FFT;

    gauge_correction = gauge_correction_FFT;

elseif run_type == run_type_poisson_ng

    update_method_title = "Second Order FFT Charge Update, BDF-1 A Update, Poisson Phi Update";
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
elseif run_type == run_type_DIRK2_ng

    update_method_title = "DIRK-2 Wave and Charge Update, FFT Derivatives, No Gauge Correction";
    update_method_folder = "DIRK2_Charge_Waves_FFT_Derivatives_ng";

    J_rho_update_method = J_rho_update_method_DIRK2;
    waves_update_method = waves_update_method_DIRK2;

    gauge_correction = gauge_correction_none;
else
    ME = MException('RunException',"No Run of " + run_type + " Type");
    throw(ME);
end

mesh_independent_variable_setup;

create_plots = @create_plots_blob;

for particle_count_multiplier = particle_count_multipliers
    for CFL = CFLs
        for g = grid_refinement
            close all;
            variable_setup;
            engine;
        end
    end
end

function [] = create_plots_blob(x, y, phi, A1, A2, rho_mesh, J1, J2, gauge_residual, gauss_residual, x1_elec_new, x2_elec_new, t, update_method_title, tag, vidObj)
    
    subplot(2,3,1);
    scatter(x1_elec_new, x2_elec_new, 5, 'filled');
    xlabel("x");
    ylabel("y");
    title("Electron Locations");
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;
    
    subplot(2,3,2);
    surf(x,y,rho_mesh);
    xlabel("x");
    ylabel("y");
    title("$\rho$",'Interpreter','latex');
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;

    % subplot(2,3,1);
    % surf(x,y,rho_mesh);
    % xlabel("x");
    % ylabel("y");
    % title("$\rho$",'Interpreter','latex');
    % xlim([x(1),x(end)]);
    % ylim([y(1),y(end)]);
    % axis square;
    % 
    % subplot(2,3,2);
    % surf(x,y,gauss_residual);
    % xlabel("x");
    % ylabel("y");
    % title("Gauss Error");
    % xlim([x(1),x(end)]);
    % ylim([y(1),y(end)]);
    % axis square;
    
    subplot(2,3,3);
    surf(x,y,gauge_residual);
    xlabel("x");
    ylabel("y");
    title("Gauge Error");
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;
    
    subplot(2,3,4);
    surf(x,y,phi(:,:,3));
    xlabel("x");
    ylabel("y");
    title("$\phi$",'Interpreter','latex');
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;
    
    subplot(2,3,5);
    surf(x,y,A1(:,:,3));
    xlabel("x");
    ylabel("y");
    title("$A_1$",'Interpreter','latex');
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;
    
    subplot(2,3,6);
    surf(x,y,A2(:,:,3));
    xlabel("x");
    ylabel("y");
    title("$A_2$",'Interpreter','latex');
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;

    
%     subplot(2,3,5);
%     surf(x,y,J1);
%     xlabel("x");
%     ylabel("y");
%     title("$J_1$",'Interpreter','latex');
%     xlim([x(1),x(end)]);
%     ylim([y(1),y(end)]);
%     axis square;
%     
%     subplot(2,3,6);
%     surf(x,y,J2);
%     xlabel("x");
%     ylabel("y");
%     title("$J_2$",'Interpreter','latex');
%     xlim([x(1),x(end)]);
%     ylim([y(1),y(end)]);
%     axis square;
    
    sgtitle({update_method_title + " method", "Grid: " + tag, "t = " + num2str(t,'%.4f')});
    
    drawnow;

    currFrame = getframe(gcf);
    writeVideo(vidObj, currFrame);
end