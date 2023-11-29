clear;
close all;

mesh_independent_variable_setup;

grid_refinement = [16,32,64];
CFLs = [1];
particle_count_multipliers = [1];

debug = false;

update_method_title = "Vanilla";
update_method_folder = "vanilla";

% update_method_title = "Non-Iterative FFT";
% update_method_folder = "non_iterative_fft";

% modification = "no_mod";
modification = "correct_gauge";

filePath = matlab.desktop.editor.getActiveFilename;
projectRoot = fileparts(filePath);

labels = string(length(grid_refinement) * length(CFLs) * length(particle_count_multipliers));

l = 1;

for particle_count_multiplier = particle_count_multipliers
    for CFL = CFLs

        for g = grid_refinement

            variable_setup;
            ts = 0:dt:T_final-dt;
            
            tag = g + "x" + g;
            path = projectRoot + "/conserving/p_mult_" + particle_count_multiplier + ...
                   "/" + tag + "/CFL_" + CFL + "/" + modification + "/" + update_method_folder + "/";
            csvPath = path + "csv_files/" + "gauge_error.csv";
            gauge_error = readmatrix(csvPath);
            
            plot(gauge_error(:,1),gauge_error(:,2));
            hold on;
            
            labels(l) = tag;
            
            l = l+1;
        end
    end
end
legend(labels);
title({"Gauge Error", update_method_title + " method, t = " + t_n, update_method_title + ", No Gauge Correction"});