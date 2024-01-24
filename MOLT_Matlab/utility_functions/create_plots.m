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

subplot(2,3,3);
surf(x,y,gauge_residual);
xlabel("x");
ylabel("y");
title("Gauge Error");
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);
axis square;

subplot(2,3,4);
surf(x,y,double(psi(:,:,3)));
xlabel("x");
ylabel("y");
title("$\phi$",'Interpreter','latex');
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);
axis square;

subplot(2,3,5);
surf(x,y,double(A1(:,:,3)));
xlabel("x");
ylabel("y");
title("$A_1$",'Interpreter','latex');
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);
axis square;

subplot(2,3,6);
surf(x,y,double(A2(:,:,3)));
xlabel("x");
ylabel("y");
title("$A_2$",'Interpreter','latex');
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);
axis square;

sgtitle({update_method_title + " method", "Grid: " + tag + ", CFL: " + CFL + ", Particle Multiplier: " + particle_count_multiplier, "t = " + num2str(t_n,'%.4f')});

drawnow;

currFrame = getframe(gcf);
writeVideo(vidObj, currFrame);

% Plotting diagnostics for Gauss' Law

% subplot(1,3,1);
% surf(x,y,ddx_E1 + ddy_E2);
% title("$\nabla \cdot \textbf{E}$",'FontSize',24);
% subplot(1,3,2);
% surf(x,y,rho_mesh/sigma_1);
% title("$\frac{\rho}{\sigma_1}$",'FontSize',24);
% subplot(1,3,3);
% surf(x,y,ddx_E1 + ddy_E2 - rho_mesh/sigma_1);
% title("$\nabla \cdot \textbf{E} - \frac{\rho}{\sigma_1}$",'FontSize',24);
% 
% subplot(3,3,1);
% surf(x,y,ddx_E1 + ddy_E2);
% title("$\nabla \cdot \textbf{E}$",'FontSize',24);
% subplot(3,3,2);
% surf(x,y,rho_mesh/sigma_1);
% title("$\frac{\rho}{\sigma_1}$",'FontSize',24);
% subplot(3,3,3);
% surf(x,y,ddx_E1 + ddy_E2 - rho_mesh/sigma_1);
% title("$\nabla \cdot \textbf{E} - \frac{\rho}{\sigma_1}$",'FontSize',24);
% 
% subplot(3,3,4);
% surf(x,y,-ddt_div_A - laplacian_phi_FFT);
% title("$-\frac{\nabla \cdot \textbf{A}^{n+1} - \nabla \cdot \textbf{A}^{n}}{\Delta t} - \Delta \phi^{n+1}$",'FontSize',24);
% subplot(3,3,5);
% surf(x,y,rho_mesh / sigma_1);
% title("$\frac{\rho^{n+1}}{\sigma_1}$",'interpreter','latex','FontSize',24);
% subplot(3,3,6);
% surf(x,y,-ddt_div_A - laplacian_phi_FFT - rho_mesh / sigma_1);
% title("$-\frac{\nabla \cdot \textbf{A}^{n+1} - \nabla \cdot \textbf{A}^{n}}{\Delta t} - \Delta \phi^{n+1} - \frac{\rho^{n+1}}{\sigma_1}$",'FontSize',24);
% 
% subplot(3,3,7);
% surf(x,y,1/kappa^2 * ddt2_phi - laplacian_phi_FFT);
% title("$\frac{1}{\kappa^2}\frac{\phi^{n+1} - 2\phi^{n} + \phi^{n-1}}{\Delta t^2} - \Delta \phi^{n+1}$",'FontSize',24);
% subplot(3,3,8);
% surf(x,y,rho_mesh / sigma_1);
% title("$\frac{\rho^{n+1}}{\sigma_1}$",'interpreter','latex','FontSize',24);
% subplot(3,3,9);
% surf(x,y,1/kappa^2 * ddt2_phi - laplacian_phi_FFT - rho_mesh / sigma_1);
% title("$\frac{1}{\kappa^2}\frac{\phi^{n+1} - 2\phi^{n} + \phi^{n-1}}{\Delta^2} - \Delta \phi^{n+1} - \frac{\rho^{n+1}}{\sigma_1}$",'FontSize',24);