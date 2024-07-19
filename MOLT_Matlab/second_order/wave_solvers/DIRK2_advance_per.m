%%
%
% Generic s=2 Butcher Table
% c1 | a11 a12
% c2 | a21 a22
% --------------
%    | b1   b2
%
% Qin and Zhang: s=2 stage, 2nd order, symplectic Diagonal Implicit RK
%
% 1/4 | 1/4   0
% 3/4 | 1/2  1/4
% --------------
%     | 1/2  1/2
%
% Pareschi and Russo: s=2 stage, 2nd order, A-stable iff x>=1/4 Diagonal Implicit RK
%
%  x  |  x    0
% 1-x | 1-2x  x
% --------------
%     | 1/2  1/2
%%
function [u_next, v_next] = DIRK2_advance_per(u, v, src, c, h, kx_deriv_2, ky_deriv_2)

    % Qin and Zhang variables
    a11 = 1/4;
    a12 = 0;
    a21 = 1/2;
    a22 = 1/4;

    b1 = 1/2;
    b2 = 1/2;

    c1 = 1/4;
    c2 = 3/4;

    % Pareschi and Russo's variables
    % x = 1/3; 
    % a11 = x;
    % a12 = 0;
    % a21 = 1 - 2*x;
    % a22 = x;
    % 
    % b1 = 1/2;
    % b2 = 1/2;
    % 
    % c1 = x;
    % c2 = 1 - x;

    alpha1 = 1/(h*a11*c);
    alpha2 = 1/(h*a22*c);

    S_prev = src(:,:,end-1);
    S_curr = src(:,:,end);

    % f^{n+c} = (1-c)f^{n} + cf^{n+1}, c in [0,1], assuming f is linear
    S_1 = (1-c1)*S_prev + c1*S_curr;
    S_2 = (1-c2)*S_prev + c2*S_curr;

    laplacian_u = compute_Laplacian_FFT(u, kx_deriv_2, ky_deriv_2);

    RHS1 = v + h*a11*c^2*(laplacian_u + S_1);
    u1 = solve_helmholtz_FFT(RHS1, alpha1, kx_deriv_2, ky_deriv_2);

    laplacian_u1 = compute_Laplacian_FFT(u1, kx_deriv_2, ky_deriv_2);

    v1 = c^2*(laplacian_u + S_1 + h*a11*laplacian_u1);

    RHS2 = v + h*a21*v1 + h*a22*c^2*(laplacian_u + h*a21*laplacian_u1 + S_2);

    u2 = solve_helmholtz_FFT(RHS2, alpha2, kx_deriv_2, ky_deriv_2);

    laplacian_u2 = compute_Laplacian_FFT(u2, kx_deriv_2, ky_deriv_2);

    v2 = c^2*(laplacian_u + h*a21*laplacian_u1 + h*a22*laplacian_u2 + S_2);

    u_next = u + h*(b1*u1 + b2*u2);
    v_next = v + h*(b1*v1 + b2*v2);

end