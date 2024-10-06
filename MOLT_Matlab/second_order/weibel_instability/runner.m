clear;
close all;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/../../utility_functions']));
addpath(genpath([fileparts(pwd), '/../../source_updaters']));
addpath(genpath([fileparts(pwd), '/../../derivatives']));
addpath(genpath([fileparts(pwd), '/../wave_solvers']));
addpath(genpath([fileparts(pwd), '/../']));

set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'cmss')
set(0,'DefaultAxesFontName', 'cmss')

grid_refinement = [32,64,16,128]; % Run FFT BDF BDF for 64x64
CFLs = [1];

debug = true;

enable_plots = true;
write_csvs = false;
plot_at = 50;
gauge_correction_none = "no_mod";
gauge_correction_FFT = "correct_gauge_fft";
gauge_correction_FD6 = "correct_gauge_fd6";

J_rho_update_method_vanilla = "vanilla";
J_rho_update_method_staggered_vanilla = "staggered-vanilla";
J_rho_update_method_BDF1_FFT = "BDF1-FFT";
J_rho_update_method_BDF2_FFT = "BDF2-FFT";
J_rho_update_method_BDF3_FFT = "BDF3-FFT";
J_rho_update_method_BDF4_FFT = "BDF4-FFT";
J_rho_update_method_CDF2_FFT = "CDF2-FFT";
J_rho_update_method_BDF1_FD6 = "BDF1-FD6";
J_rho_update_method_BDF2_FD6 = "BDF2-FD6";
J_rho_update_method_BDF1_FD8 = "BDF1-FD8";
J_rho_update_method_BDF2_FD8 = "BDF2-FD8";
J_rho_update_method_DIRK2 = "DIRK2";
J_rho_update_method_FD2 = "FD2";
J_rho_update_method_FD4 = "FD4";
J_rho_update_method_FD6 = "FD6";

waves_update_method_BDF1_MOLT_Pure = "BDF1-MOLT-PURE";
waves_update_method_BDF2_MOLT_Pure = "BDF2-MOLT-PURE";
waves_update_method_CDF2_implicit_MOLT_Pure = "CDF2-implicit-MOLT-PURE";
waves_update_method_CDF2_semi_implicit_MOLT_Pure = "CDF2-semi-implicit-MOLT-PURE";
waves_update_method_FFT = "NA";
waves_update_method_BDF1_MOLT_Hybrid = "BDF1-MOLT";
waves_update_method_BDF2_MOLT_Hybrid = "BDF2-MOLT";
waves_update_method_BDF3_MOLT_Hybrid = "BDF3-MOLT";
waves_update_method_BDF4_MOLT_Hybrid = "BDF4-MOLT";
waves_update_method_BDF1_FFT = "BDF1-FFT";
waves_update_method_BDF2_FFT = "BDF2-FFT";
waves_update_method_BDF3_FFT = "BDF3-FFT";
waves_update_method_BDF4_FFT = "BDF4-FFT";
waves_update_method_CDF2_FFT = "CDF2-FFT";

waves_update_method_BDF1_FD6 = "BDF1-FD6";
waves_update_method_BDF2_FD6 = "BDF2-FD6";
waves_update_method_CDF2_FD6 = "CDF2-FD6";
waves_update_method_BDF1_FD8 = "BDF1-FD8";
waves_update_method_BDF2_FD8 = "BDF2-FD8";
waves_update_method_CDF2_FD8 = "CDF2-FD8";
waves_update_method_DIRK2 = "DIRK2";
waves_update_method_poisson_phi = "poisson_phi";

waves_update_method_pure_FFT = "pure_fft";

run_type_vanilla_ng = "vanilla_no_gauge_cleaning";
run_type_vanilla_gc = "vanilla_gauge_cleaning";

run_type_FFT_BDF1_ng = "BDF1_FFT_no_gauge_cleaning";
run_type_FFT_BDF1_gc = "BDF1_FFT_gauge_cleaning";
run_type_FFT_BDF2_ng = "BDF2_FFT_no_gauge_cleaning";
run_type_FFT_BDF2_gc = "BDF2_FFT_gauge_cleaning";

run_type_FD6_BDF1_ng = "BDF1_FD6_no_gauge_cleaning";
run_type_FD6_BDF1_gc = "BDF1_FD6_gauge_cleaning";
run_type_FD6_BDF2_ng = "BDF2_FD6_no_gauge_cleaning";
run_type_FD6_BDF2_gc = "BDF2_FD6_gauge_cleaning";
run_type_FD6_CDF2_ng = "CDF2_FD6_no_gauge_cleaning";
run_type_FD6_CDF2_gc = "CDF2_FD6_gauge_cleaning";

run_type_FD8_BDF1_ng = "BDF1_FD8_no_gauge_cleaning";
run_type_FD8_BDF1_gc = "BDF1_FD8_gauge_cleaning";
run_type_FD8_BDF2_ng = "BDF2_FD8_no_gauge_cleaning";
run_type_FD8_BDF2_gc = "BDF2_FD8_gauge_cleaning";
run_type_FD8_CDF2_ng = "CDF2_FD8_no_gauge_cleaning";
run_type_FD8_CDF2_gc = "CDF2_FD8_gauge_cleaning";

run_type_MOLT_Pure_BDF1_ng = "BDF1_MOLT_PURE_no_gauge_cleaning";
run_type_MOLT_Pure_BDF2_ng = "BDF2_MOLT_PURE_no_gauge_cleaning";

run_type_MOLT_Pure_CDF2_semi_implicit_ng = "CDF2_MOLT_PURE_semi_implicit_no_gauge_cleaning";
run_type_MOLT_Pure_CDF2_implicit_ng = "CDF2_MOLT_PURE_implicit_no_gauge_cleaning";

run_type_MOLT_Pure_BDF1_gc = "BDF1_MOLT_PURE_gauge_cleaning";
run_type_MOLT_Pure_BDF2_gc = "BDF2_MOLT_PURE_gauge_cleaning";
run_type_MOLT_Pure_CDF2_gc = "BDF2_MOLT_PURE_gauge_cleaning";

run_type_FFT_CDF2_ng = "CDF2_FFT_no_gauge_cleaning";
run_type_vanilla_J_rho_CDF2_wave_ng = "CDF2_vanilla_J_rho_FFT_waves_no_gauge_cleaning";

run_type_FFT_CDF2_ng = "CDF2_FFT_no_gauge_cleaning";

run_type_pure_FFT_ng = "pure_FFT_no_gauge_cleaning";
run_type_pure_FFT_gc = "pure_FFT_gauge_cleaning";

run_type_poisson_ng = "FFT_A_poisson_phi_no_gauge_cleaning";

run_type_DIRK2_ng = "DIRK_FFT_deriv_no_gauge_cleaning";

J_rho_BDF_FFT_Family = [J_rho_update_method_BDF1_FFT, J_rho_update_method_BDF2_FFT, J_rho_update_method_BDF3_FFT, J_rho_update_method_BDF4_FFT, J_rho_update_method_CDF2_FFT];
waves_BDF_FFT_Family = [waves_update_method_BDF1_FFT, waves_update_method_BDF2_FFT, waves_update_method_BDF3_FFT, waves_update_method_BDF4_FFT, waves_update_method_CDF2_FFT];

J_rho_BDF_FD6_Family = [J_rho_update_method_BDF1_FD6, J_rho_update_method_BDF2_FD6];
waves_BDF_FD6_Family = [waves_update_method_BDF1_FD6, waves_update_method_BDF2_FD6];

J_rho_BDF_FD8_Family = [J_rho_update_method_BDF1_FD8, J_rho_update_method_BDF2_FD8];
waves_BDF_FD8_Family = [waves_update_method_BDF1_FD8, waves_update_method_BDF2_FD8];

% MOLT for the wave, FFT for the derivatives
waves_BDF_Hybrid_Family = [waves_update_method_BDF1_MOLT_Hybrid, waves_update_method_BDF2_MOLT_Hybrid, waves_update_method_BDF3_MOLT_Hybrid, waves_update_method_BDF4_MOLT_Hybrid];
waves_CDF_Hybrid_Family = [waves_update_method_CDF2_implicit_MOLT_Pure, waves_update_method_CDF2_semi_implicit_MOLT_Pure, waves_update_method_CDF2_FFT, waves_update_method_CDF2_FD6];

for g = grid_refinement
    for iter = 6:6
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % THIS IS THE ONLY PARAMETER TO TWEAK
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % run_type = run_type_vanilla_ng;
        % run_type = run_type_pure_FFT_ng;
        % run_type = run_type_FFT_BDF1_ng;
        % run_type = run_type_FFT_BDF2_ng;
        % run_type = run_type_poisson_ng;
        % run_type = run_type_DIRK2_ng;
        
        if iter == 1
            run_type = run_type_FFT_BDF1_ng;
        elseif iter == 2
            run_type = run_type_FFT_BDF2_ng;
        elseif iter == 3
            run_type = run_type_FD6_BDF1_ng;
        elseif iter == 4
            run_type = run_type_FD6_BDF2_ng;
        elseif iter == 5
            run_type = run_type_FD8_BDF1_ng;
        elseif iter == 6
            run_type = run_type_FD8_BDF2_ng;
        elseif iter == 7
            run_type = run_type_DIRK2_ng;
        elseif iter == 8
            run_type = run_type_FFT_CDF2_ng;
        elseif iter == 9
            run_type = run_type_vanilla_J_rho_CDF2_wave_ng;
        elseif iter == 10
            run_type = run_type_MOLT_Pure_BDF1_ng;
        elseif iter == 11
            run_type = run_type_MOLT_Pure_BDF2_ng;
        elseif iter == 12
            run_type = run_type_MOLT_Pure_CDF2_implicit_ng;
        elseif iter == 13
            run_type = run_type_MOLT_Pure_CDF2_semi_implicit_ng;
        elseif iter == 14
            run_type = run_type_FD6_CDF2_ng;
        end
    
        if run_type == run_type_MOLT_Pure_BDF1_ng
        
            update_method_title = "Second Order Vanilla Charge Update, BDF-1 Wave Update, MOLT Wave Solver";
            update_method_folder = "vanilla_charge_BDF1_wave_update";
        
            J_rho_update_method = J_rho_update_method_vanilla;
            waves_update_method = waves_update_method_BDF1_MOLT_Pure;
        
            gauge_correction = gauge_correction_none;
    
        elseif run_type == run_type_MOLT_Pure_BDF1_gc
        
            update_method_title = "Second Order Vanilla Charge Update, BDF-1 Wave Update, MOLT Wave Solver";
            update_method_folder = "vanilla_charge_BDF1_wave_update";
        
            J_rho_update_method = J_rho_update_method_vanilla;
            waves_update_method = waves_update_method_BDF1_MOLT_Pure;
        
            gauge_correction = gauge_correction_FFT;
    
        elseif run_type == run_type_MOLT_Pure_BDF2_ng
        
            update_method_title = "Second Order Vanilla Charge Update, BDF-2 Wave Update, MOLT Wave Solver";
            update_method_folder = "vanilla_charge_BDF2_wave_update";
        
            J_rho_update_method = J_rho_update_method_vanilla;
            waves_update_method = waves_update_method_BDF2_MOLT_Pure;
        
            gauge_correction = gauge_correction_none;
    
        elseif run_type == run_type_MOLT_Pure_CDF2_implicit_ng
        
            update_method_title = "Second Order Vanilla Charge Update, Implicit CDF-2 Wave Update, MOLT Wave Solver";
            update_method_folder = "vanilla_charge_CDF2_implicit_wave_update";
        
            J_rho_update_method = J_rho_update_method_vanilla;
            waves_update_method = waves_update_method_CDF2_implicit_MOLT_Pure;
        
            gauge_correction = gauge_correction_none;
    
        elseif run_type == run_type_MOLT_Pure_CDF2_semi_implicit_ng
        
            update_method_title = "Second Order Vanilla Charge Update, Semi-Implicit CDF-2 Wave Update, MOLT Wave Solver";
            update_method_folder = "vanilla_charge_CDF2_semi_implicit_wave_update";
        
            J_rho_update_method = J_rho_update_method_vanilla;
            waves_update_method = waves_update_method_CDF2_semi_implicit_MOLT_Pure;
        
            gauge_correction = gauge_correction_none;
        
        elseif run_type == run_type_FD6_BDF1_ng
        
            update_method_title = "Second Order FD6 Charge Update, BDF-1 Wave Update, FD6 Derivative";
            update_method_folder = "FD6_charge_BDF1_wave_update_FD6_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF1_FD6;
            waves_update_method = waves_update_method_BDF1_FD6;
        
            gauge_correction = gauge_correction_none;
        
        elseif run_type == run_type_FD6_BDF2_ng
        
            update_method_title = "Second Order FD6 Charge Update, BDF-2 Wave Update, FD6 Derivative";
            update_method_folder = "FD6_charge_BDF2_wave_update_FD6_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF2_FD6;
            waves_update_method = waves_update_method_BDF2_FD6;
        
            gauge_correction = gauge_correction_none;
        
        elseif run_type == run_type_FD8_BDF1_ng
        
            update_method_title = "Second Order FD8 Charge Update, BDF-1 Wave Update, FD8 Derivative";
            update_method_folder = "FD8_charge_BDF1_wave_update_FD8_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF1_FD8;
            waves_update_method = waves_update_method_BDF1_FD8;
        
            gauge_correction = gauge_correction_none;
        
        elseif run_type == run_type_FD8_BDF2_ng
        
            update_method_title = "Second Order FD8 Charge Update, BDF-2 Wave Update, FD8 Derivative";
            update_method_folder = "FD8_charge_BDF2_wave_update_FD8_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF2_FD8;
            waves_update_method = waves_update_method_BDF2_FD8;
        
            gauge_correction = gauge_correction_none;
        
        elseif run_type == run_type_FD6_CDF2_ng
        
            update_method_title = "Second Order FD6 Charge Update, CDF-2 Wave Update, FD6 Derivative";
            update_method_folder = "FD6_charge_CDF2_wave_update_FD6_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF1_FD6;
            waves_update_method = waves_update_method_CDF2_FD6;
        
            gauge_correction = gauge_correction_none;
        
        elseif run_type == run_type_FFT_BDF1_ng
        
            update_method_title = "Second Order FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
            update_method_folder = "FFT_charge_BDF1_wave_update_FFT_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF1_FFT;
            waves_update_method = waves_update_method_BDF1_FFT;
        
            gauge_correction = gauge_correction_none;
        
        elseif run_type == run_type_FFT_BDF1_gc
        
            update_method_title = "Second Order FFT Charge Update, BDF-1 Wave Update, FFT Derivative";
            update_method_folder = "FFT_charge_BDF1_wave_update_FFT_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF1_FFT;
            waves_update_method = waves_update_method_BDF1_FFT;
        
            gauge_correction = gauge_correction_FFT;
        
        elseif run_type == run_type_FFT_BDF2_ng
        
            update_method_title = "Second Order FFT Charge Update, BDF-2 Wave Update, FFT Derivative";
            update_method_folder = "FFT_charge_BDF2_wave_update_FFT_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF2_FFT;
            waves_update_method = waves_update_method_BDF2_FFT;
        
            gauge_correction = gauge_correction_none;
        
        elseif run_type == run_type_FFT_BDF2_gc
        
            update_method_title = "Second Order FFT Charge Update, BDF-2 Wave Update, FFT Derivative";
            update_method_folder = "FFT_charge_BDF2_wave_update_FFT_derivative";
        
            J_rho_update_method = J_rho_update_method_BDF2_FFT;
            waves_update_method = waves_update_method_BDF2_FFT;
        
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
        elseif run_type == run_type_FFT_CDF2_ng
    
            update_method_title = "Second Order FFT Charge Update, BDF-1 Wave Update, Time Centered, FFT Derivative";
            update_method_folder = "FFT_charge_CDF2_wave_update_FFT_derivative";
        
            J_rho_update_method = J_rho_update_method_CDF2_FFT;
            waves_update_method = waves_update_method_CDF2_FFT;
        
            gauge_correction = gauge_correction_none;
        elseif run_type == run_type_vanilla_J_rho_CDF2_wave_ng
    
            update_method_title = "Vanilla Charge Update, BDF-1 Wave Update, Time Centered, FFT Derivative";
            update_method_folder = "vanilla_charge_CDF2_wave_update_FFT_derivative";
        
            J_rho_update_method = J_rho_update_method_staggered_vanilla;
            waves_update_method = waves_update_method_CDF2_FFT;
        
            gauge_correction = gauge_correction_none;
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
        
        create_plots = @create_plots_weibel;
    
        for CFL = CFLs
            close all;
            variable_setup;
            % make_vpa;
            engine;
        end
    end
end

function [] = create_plots_weibel(x, y, phi, A1, A2, E1, E2, B3, rho_mesh, J1, J2, gauge_residual, gauss_residual, x1_elec, x2_elec, t, update_method_title, tag, vidObj)
    
    subplot(2,3,1);
    surf(x,y,rho_mesh);
    xlabel("x");
    ylabel("y");
    title("$\rho$",'Interpreter','latex');
    axis square;
    
    subplot(2,3,2);
    surf(x,y,J1);
    xlabel("x");
    ylabel("y");
    title("$J_1$",'Interpreter','latex');
    axis square;
    
    subplot(2,3,3);
    surf(x,y,J2);
    xlabel("x");
    ylabel("y");
    title("$J_2$",'Interpreter','latex');
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;
    
    subplot(2,3,4);
    surf(x,y,E1);
    xlabel("x");
    ylabel("y");
    title("$E_x$",'Interpreter','latex');
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;
    
    subplot(2,3,5);
    surf(x,y,E2);
    xlabel("x");
    ylabel("y");
    title("$E_y$",'Interpreter','latex');
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    axis square;
    
    subplot(2,3,6);
    surf(x,y,B3);
    xlabel("x");
    ylabel("y");
    title("$B_z$",'Interpreter','latex');
    xlim([x(1),x(end)]);
    ylim([y(1),y(end)]);
    view(2);
    shading interp;
    colorbar;
    axis square;
    
    sgtitle({update_method_title + " method", "Grid: " + tag, "t = " + num2str(t,'%.4f')});
    
    drawnow;

    currFrame = getframe(gcf);
    writeVideo(vidObj, currFrame);
end