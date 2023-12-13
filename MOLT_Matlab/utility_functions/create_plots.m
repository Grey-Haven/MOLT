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