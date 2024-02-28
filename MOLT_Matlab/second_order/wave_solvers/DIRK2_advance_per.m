%%
% Qin and Zhang two-stage, 2nd order, symplectic DIRK method:
% 
% 1/4 | 1/4   0
% 3/4 | 1/2  1/4
% ---------------
%     | 1/2  1/2
%
%%

function [u_next,v_next] = DIRK2_advance_per(u, v, src_data, c, h, kx_deriv_2, ky_deriv_2)
    
    % Assuming the source function from S^{n} to S^{n+1} is a linear
    % update, we have S^{n+a} = (1-a)S^{n} + aS^{n+1}.

    c1 = 1/4;
    c2 = 3/4;

    S_1 = (1 - c1) * src_data(:,:,end-1) + c1 * src_data(:,:,end);
    S_2 = (1 - c2) * src_data(:,:,end-1) + c2 * src_data(:,:,end);

    a_11 = 1/4;
    a_12 = 0;
    a_21 = 1/2;
    a_22 = 1/4;

    b_1 = 1/2;
    b_2 = 1/2;

    alpha_11 = 1/(h*a_11*c);
    alpha_22 = 1/(h*a_22*c);
    
    laplacian_u = compute_Laplacian_FFT(u,kx_deriv_2,ky_deriv_2);
    % laplacian_v = compute_Laplacian_FFT(v,kx_deriv_2,ky_deriv_2);

    u_1 = solve_helmholtz_FFT(v + h*a_11*c^2*(laplacian_u + S_1), alpha_11, kx_deriv_2, ky_deriv_2);

    laplacian_u1 = compute_Laplacian_FFT(u_1,kx_deriv_2,ky_deriv_2);

    v_1 = c^2*(laplacian_u + S_1 + h*a_11*laplacian_u1);

    u_2 = solve_helmholtz_FFT(v + h*a_21*v_1 + h*a_22*c^2*(laplacian_u + h*a_21*laplacian_u1 + S_2), alpha_22, kx_deriv_2, ky_deriv_2);

    laplacian_u2 = compute_Laplacian_FFT(u_2,kx_deriv_2,ky_deriv_2);

    v_2 = c^2*(laplacian_u + a_21*h*laplacian_u1 + a_22*h*laplacian_u2 + S_2);

    v_next = v + h*(b_1*v_1 + b_2*v_2);
    u_next = u + h*(b_1*u_1 + b_2*u_2);
end