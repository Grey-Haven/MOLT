figure(2);
ts = dt*(1:steps);
shift = 2e-4;

growth_rate_comparison3 = 3e-1;
growth_rate_comparison31 = 3.1e-1;
growth_rate_comparison4 = 4e-1;

max_compare_step = min(steps,16000);
growth_comparison3 = shift*exp(growth_rate_comparison3*ts(1:max_compare_step));
growth_comparison31 = shift*exp(growth_rate_comparison31*ts(1:max_compare_step));
growth_comparison4 = shift*exp(growth_rate_comparison4*ts(1:max_compare_step));
semilogy(ts,Bz_L2_hist(1:steps));
hold on;
semilogy(ts(1:max_compare_step),growth_comparison3);
semilogy(ts(1:max_compare_step),growth_comparison31);
semilogy(ts(1:max_compare_step),growth_comparison4);
hold off;
legend("Numerical", "Growth Rate = .3", "Growth Rate = .31", "Growth Rate = .4");
xlabel("Angular Plasma Periods");
ylabel("||B^{(3)}||_2");
title(update_method_title + " method");
drawnow;