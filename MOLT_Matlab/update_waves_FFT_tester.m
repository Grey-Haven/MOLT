close all;
clear;
addpath(genpath([fileparts(pwd)]));
addpath(genpath([fileparts(pwd), '/utility_functions']));
addpath(genpath([fileparts(pwd), '/rho_updaters']));

N = 32;
N_x = N+1;
N_y = N+1;

psi = zeros(N_y,N_x,3);
A1 = zeros(N_y,N_x,3);
A2 = zeros(N_y,N_x,3);

ddx_psi = zeros(N_y,N_x);
ddy_psi = zeros(N_y,N_x);
psi_src = zeros(N_y,N_x);

ddx_A1 = zeros(N_y,N_x);
ddy_A1 = zeros(N_y,N_x);
A1_src = zeros(N_y,N_x);

ddx_A2 = zeros(N_y,N_x);
ddy_A2 = zeros(N_y,N_x);
A2_src = zeros(N_y,N_x);

L_x = 1.0;
L_y = 1.0;

a_x = -L_x/2;
b_x = L_x/2;

a_y = -L_y/2;
b_y =  L_y/2;

dx = (b_x - a_x)/(N_x - 1);
dy = (b_y - a_y)/(N_y - 1);

x = linspace(a_x, b_x, N_x);
y = linspace(a_y, b_y, N_y);

x_star = x(1:end-1);
y_star = y(1:end-1);

% kx = 2*pi/(L_x)*[0:(N_x-1)/2-1, 0, -(N_x-1)/2+1:-1];
% ky = 2*pi/(L_y)*[0:(N_y-1)/2-1, 0, -(N_y-1)/2+1:-1];
% Calculate wavenumber
dkx = 1 / ((N_x-1) * dx); % Wavenumber increment in x direction
dky = 1 / ((N_y-1) * dy); % Wavenumber increment in y direction
kx = 2*pi*fftshift((-(N_x-1)/2:(N_x-1)/2-1) * dkx);
ky = 2*pi*fftshift((-(N_y-1)/2:(N_y-1)/2-1) * dky);

xi_x = 2*pi;
xi_y = 2*pi;
% xi_t = 2*pi;
xi_t = sqrt(xi_x^2 + xi_y^2);
kappa = .1;

test_func = sin(xi_y*y_star)'.*sin(xi_x*x_star);
test_func_xx = -xi_x^2*sin(xi_y*y_star)'.*sin(xi_x*x_star);
test_func_yy = -xi_y^2*sin(xi_y*y_star)'.*sin(xi_x*x_star);

test_func_fft_xx = ifft(-kx.^2.*fft(test_func,N_x-1,2),N_x-1,2);
test_func_fft_yy = ifft(-ky'.^2.*fft(test_func,N_y-1,1),N_y-1,1);

assert(norm(test_func_fft_xx - test_func_xx) < 1e-10);

T = 4*pi;

t = 0;
dt = .001;


psi(:,:,1) = u(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
psi(:,:,2) = u(x,y,t,xi_x,xi_y,xi_t,kappa);

A1(:,:,1) = u(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
A1(:,:,2) = u(x,y,t,xi_x,xi_y,xi_t,kappa);

A2(:,:,1) = u(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
A2(:,:,2) = u(x,y,t,xi_x,xi_y,xi_t,kappa);

t = dt;

psi_err = zeros(floor(T/dt),1);
A1_err = zeros(floor(T/dt),1);
A2_err = zeros(floor(T/dt),1);

step = 1;
plot_at = 250;

while t < T
    
    psi_src = S(x,y,t,xi_x,xi_y,xi_t,kappa);
    A1_src = S(x,y,t-dt,xi_x,xi_y,xi_t,kappa);
    A2_src = S(x,y,t-dt,xi_x,xi_y,xi_t,kappa);

    update_waves_FFT_alt;
    
    psi = shuffle_steps(psi);
    A1 = shuffle_steps(A1);
    A2 = shuffle_steps(A2);
    analytic = u(x,y,t,xi_x,xi_y,xi_t,kappa);
    
    if mod(step,plot_at) == 0
        
        subplot(1,3,1);
        surf(x,y,A2(:,:,3));
        xlabel("x");
        ylabel("y");
        title("Approx A2");
        zlim([-1,1]);

        subplot(1,3,2);
        surf(x,y,u(x,y,t,xi_x,xi_y,xi_t,kappa));
        xlabel("x");
        ylabel("y");
        title("Analytic");
        zlim([-1,1]);

        subplot(1,3,3);
        surf(x,y,u(x,y,t,xi_x,xi_y,xi_t,kappa) - A2(:,:,3));
        xlabel("x");
        ylabel("y");
        title("Analytic - Approx Psi");
%         zlim([-1e-2,1e-2]);

        sgtitle("t = " + t);
        drawnow;
    end

    psi_err(step) = norm(analytic - psi(:,:,3),"inf");
    A1_err(step) = norm(analytic - A1(:,:,3),"inf");
    A2_err(step) = norm(analytic - A2(:,:,3),"inf");
    
    t = t + dt;
    step = step + 1;
end

ts = 0:dt:T-dt;
plot(ts,psi_err);
hold on;
plot(ts,A1_err);
plot(ts,A2_err);
xlabel("t");
ylabel("Inf Err");
legend("Psi", "A_1", "A_2");



function src = S(x,y,t, xi_x,xi_y,xi_t,k)
    src = -(xi_x^2 + xi_y^2 - xi_t^2) * u(x,y,t,xi_x,xi_y,xi_t,k);
end

function analytic = u(x,y,t, xi_x,xi_y,xi_t,k)
    analytic = cos(xi_y*y)' .* cos(xi_x*x) * sin(k*xi_t*t);
end