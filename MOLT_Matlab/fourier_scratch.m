close all;
clear;

N = 64;

Nx = N;
Ny = N;

ax = 0;
bx = 1;
ay = 0;
by = 1;

Lx = bx - ax;
Ly = by - ay;

dx = Lx / Nx;
dy = Ly / Ny;

x = ax:dx:bx-dx;
y = ay:dy:by-dy;

kx = 2*pi/(bx-ax)*[0:N/2-1, 0, -N/2+1:-1];
ky = 2*pi/(by-ay)*[0:N/2-1, 0, -N/2+1:-1];

n = length(x);
freq = 1/dx;
fshift = (-n/2:n/2-1)*(freq/n);

f = sin(2*pi*x);
df = 2*pi*cos(2*pi*x);
ff = fft(f);
subplot(3,1,1);
plot(x,f);
subplot(3,1,2);
plot(x,df);
title("Analytic Derivative");
subplot(3,1,3);
plot(x,ifft(sqrt(-1)*kx.*ff));
title("Fourier Derivative");

J = sin(8*pi*y)'*sin(2*pi*x);
Jx_deriv = 2*pi*sin(8*pi*y)'*cos(2*pi*x);
Jy_deriv = 8*pi*cos(8*pi*y)'*sin(2*pi*x);

FFT_x_J = fft(J);
FFT_y_J = fft(J,n,2);
FFT_xy_J = fft(FFT_x_J,n,2);
FFT_yx_J = fft(FFT_y_J);

FFT_x_J = fftshift(FFT_x_J);
FFT_y_J = fftshift(FFT_y_J);
FFT_xy_J = fftshift(FFT_xy_J);
FFT_yx_J = fftshift(FFT_yx_J);

assert(norm(abs(FFT_xy_J - FFT_yx_J)) < 1e-10);

subplot(2,2,1);
surf(x,y,J')
xlabel("x")
ylabel("y")
title("J")
subplot(2,2,2);
surf(fshift,y,abs(FFT_x_J)');
xlabel("k_x")
ylabel("y")
title("fft_x(J)")
subplot(2,2,3);
surf(x,fshift,abs(FFT_y_J)');
xlabel("x")
ylabel("k_y")
title("fft_{y}(J)");
subplot(2,2,4);
surf(fshift,fshift,abs(FFT_xy_J)');
xlabel("k_x")
ylabel("k_y")
title("fft_{xy}(J)");


Jx_deriv_star = ifft(sqrt(-1)*kx.*fft(J,n,2),n,2);
Jy_deriv_star = ifft(sqrt(-1)*ky'.*fft(J,n,1),n,1);


figure;
subplot(2,2,1);
surf(x,y,Jx_deriv')
xlabel("x")
ylabel("y")
title("J_x")
subplot(2,2,2);
surf(x,y,Jx_deriv_star');
xlabel("x")
ylabel("y")
title("J_x^*")
subplot(2,2,3);
surf(x,fshift,Jy_deriv');
xlabel("x")
ylabel("y")
title("J_y");
subplot(2,2,4);
surf(x,y,Jy_deriv_star');
xlabel("x")
ylabel("y")
title("J_y^*");