subplot(2,3,1);
surf(x,y,rho_mesh);
xlabel("x");
ylabel("y");
title("$\rho$",'Interpreter','latex');
axis square;

subplot(2,3,2);
surf(x,y,J_mesh(:,:,1));
xlabel("x");
ylabel("y");
title("$J_1$",'Interpreter','latex');
axis square;

subplot(2,3,3);
surf(x,y,J_mesh(:,:,2));
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