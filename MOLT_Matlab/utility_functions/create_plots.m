subplot(2,2,1);
scatter(x1_elec_new, x2_elec_new, 5, 'filled');
xlabel("x");
ylabel("y");
title("Electron Locations");
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);

subplot(2,2,2);
surf(x,y,rho_mesh);
xlabel("x");
ylabel("y");
title("$\rho$",'Interpreter','latex');
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);

subplot(2,2,3);
surf(x,y,A2(:,:,3));
xlabel("x");
ylabel("y");
title("$A_2$",'Interpreter','latex');
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);

subplot(2,2,4);
surf(x,y,gauge_residual);
xlabel("x");
ylabel("y");
title("Gauge Error");
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);

sgtitle("FFT Iterative method, t = " + t_n);

drawnow;

currFrame = getframe(gcf);
writeVideo(vidObj, currFrame);