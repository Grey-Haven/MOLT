clear;
close all;

grids = [8,16,32,64,128,256];
errs = zeros(length(grids),1);
dxs = zeros(length(grids),1);
i = 1;

x_eq = @(x_arg) x_arg + sin(x_arg);
y_eq = @(y_arg) ones(length(y_arg));

x_deriv = @(x_arg) 1 + cos(x_arg);
y_deriv = @(y_arg) zeros(length(y_arg));

N=256;
L=2*pi;
dx=L/N;
x=linspace(-L/2,L/2-dx,N);
% f = exp(-x.^2);
f = (x/pi).^2 - 1;
k(1:N/2)=1i*(0:N/2-1);
k(N/2+2:N)=1i*(-N/2+1:-1);
% Consider the Nyquist mode my young Padawan
k(N/2+1)=0;
fhat = fft(f);
f_fft = ifft(fhat);
fhatx=k.*fhat;
fx_fft=ifft(fhatx);
% fx = -2*x.*exp(-x.^2);
fx = 2*x / pi^2;

subplot(1,2,1);
plot(x,f);
hold on;
plot(x,f_fft);
legend(["Original", "F^{-1}[F[f]]"]);
title("f(x) = (x/pi)^2 - 1");
hold off;
subplot(1,2,2);
plot(x,fx);
hold on;
plot(x,fx_fft);
legend(["Analytic", "FFT"]);
title("f'(x) = (2/pi^2)x");
hold off;

for g = grids
    N_x = g+1;
    N_y = g+1;
    
    a_x = -10;
    a_y = -10;
    b_x =  10;
    b_y =  10;
    
    L_x = b_x - a_x;
    L_y = b_y - a_y;
    
    dx = L_x / (N_x - 1);
    dy = L_y / (N_y - 1);
    
    x = a_x:dx:b_x;
    y = a_y:dy:b_y;
    
    kx_deriv_1 = 2*pi/(L_x)*[0:(N_x-1)/2-1, 0, -(N_x-1)/2+1:-1];
    ky_deriv_1 = 2*pi/(L_y)*[0:(N_y-1)/2-1, 0, -(N_y-1)/2+1:-1];
    kx_deriv_2 = 2*pi/(L_x)*[0:(N_x-1)/2-1, -(N_x)/2, -(N_x-1)/2+1:-1];
    ky_deriv_2 = 2*pi/(L_y)*[0:(N_y-1)/2-1, -(N_y)/2, -(N_y-1)/2+1:-1];
    
    u = x_eq(x)'.*y_eq(y);
    u_ddx = x_deriv(x)'.*y_eq(y);
    u_ddy = x_eq(x)'.*y_deriv(y);

    u_ddx_fft = compute_ddx_FFT_dir(u,kx_deriv_1);
    u_ddy_fft = compute_ddy_FFT_dir(u,ky_deriv_1);

    errs(i) = max(max(abs(u_ddx - u_ddx_fft)));
    dxs(i) = dx;
    i = i+1;
end

loglog(dxs,errs);