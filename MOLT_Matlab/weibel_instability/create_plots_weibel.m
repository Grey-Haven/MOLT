subplot(2,3,1);
scatter_skip = 1;
scatter3(x1_elec_new(1:scatter_skip:N_p), x2_elec_new(1:scatter_skip:N_p), ones(length(x2_elec_new(1:scatter_skip:N_p)),1), 5, 'filled', 'MarkerFaceColor', [0,0,1]);
hold on;
scatter3(x1_elec_new(N_p+1:scatter_skip:end), x2_elec_new(N_p+1:scatter_skip:end), -ones(length(x2_elec_new(N_p+1:scatter_skip:end)),1), 5, 'filled', 'MarkerFaceColor', [1,0,0]);
hold off;
xlabel("x");
ylabel("y");
zlabel("Initial Distribution")
title("Electron Locations");
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);
axis square;

subplot(2,3,2);
scatter(v1_elec_new, v2_elec_new, 5, 'filled');
xlabel("v_x");
ylabel("v_y");
title("Electron Velocities");
xlim([-10*v1_therm,10*v1_therm]);
ylim([-10*v2_therm,10*v2_therm]);
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


% subplot(2,3,5);
% surf(x,y,psi_A);
% xlabel("x");
% ylabel("y");
% title("$\phi_A$",'Interpreter','latex');
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);
% 
% subplot(2,3,6);
% surf(x,y,psi_C);
% xlabel("x");
% ylabel("y");
% title("$\phi_C$",'Interpreter','latex');
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);

sgtitle({update_method_title + " method", "Grid: " + tag + ", CFL: " + CFL + ", Particle Multiplier: " + particle_count_multiplier, "t = " + num2str(t_n,'%.4f')});

drawnow;

currFrame = getframe(gcf);
writeVideo(vidObj, currFrame);