clear;
close all force;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));
addpath(genpath([fileparts(pwd), '/wave_solvers']));

set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'cmss')
set(0,'DefaultAxesFontName', 'cmss')

grid_refinement = [128];
CFLs = [.125,.25,.5,1,2,4,8];

debug = true;

enable_plots = true;
write_csvs = false;
plot_at = 50;


lgd = legend('show', 'Location', 'southeast', 'FontSize', 14);

lgd.Position = lgd.Position + [-0.05 0.1 0 0];

% Number of grid points and cell spacing

mesh_independent_variable_setup;
create_plots = @create_plots_twostream;
for g = grid_refinement

    figure;
    x0=200;
    y0=100;
    width = 1200;
    height = 1200;
    set(gcf,'position',[x0,y0,width,height]);

    for CFL = CFLs
        variable_setup;
        engine_ES_1D;
    end

    % plot(ts,1e-4*exp(.276*ts),'--','color','k');
    % ylim([1e-6,1e2]);
    % lgd = legend(["CFL 1", "CFL 2", "CFL 4", "CFL 8", "Analytical"], 'Location', 'southeast', 'FontSize', 14);
    % lgd.Position = lgd.Position + [-0.05 0.1 0 0];
    % 
    % xlabel("Angular Plasma Periods",'FontSize',24,'Interpreter','latex');
    % ylabel("$\log{\left(||E(x)||_2\right)}$",'FontSize',24,'Interpreter','latex');
    % title("Two Stream Instability", 'FontSize', 48);
    % saveas(gcf, "./results/" + tag + "/electric_magnitude.jpg");
    % close all;
end

function [] = create_plots_twostream(x, x1, v1, phi, rho, t, tag, CFL, vidObj)
    
    s = length(x1) / 2;

    % subplot(1,3,1);
    scatter(x1(1:s), v1(1:s), 5, 'filled');
    hold on;
    scatter(x1(s-1:end), v1(s-1:end), 5, 'filled');
    hold off;
    xlabel("x");
    ylabel("v");
    xlim([x(1),x(end)]);
    ylim([-3,3]);

    % subplot(1,3,2);
    % plot(x, phi);
    % xlabel("x");
    % ylabel("$\phi$", 'Interpreter', 'latex');
    % title("$\phi$", 'Interpreter', 'latex');
    % 
    % subplot(1,3,3);
    % plot(x, rho);
    % xlabel("x");
    % ylabel("$\rho$", 'Interpreter', 'latex');
    % title("$\rho$", 'Interpreter', 'latex');
    
    sgtitle({"Phase Space " "t = " + num2str(t,'%.4f'), "Grid: " + tag + ", CFL = " + CFL});
    
    drawnow;

    currFrame = getframe(gcf);
    writeVideo(vidObj, currFrame);
end