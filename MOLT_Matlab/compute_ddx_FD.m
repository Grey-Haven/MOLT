function [] = compute_ddx_FD(dudx, u, dx)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Computes an x derivative via finite differences
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    N_x = size(u,1);
    N_y = size(u,2);

    % Left boundary (forward diff)
    for j = 1:N_y
        dudx(1,j) = ( -3*u(1,j) + 4*u(2,j) - u(3,j) )/(2*dx);
    end 
    % Central derivatives
    for j = 1:N_y
        for i = 2:N_x-1
            dudx(i,j) = ( u(i+1,j) - u(i-1,j) )/(2*dx);
        end
    end
    % Right boundary (backward diff)
    for j = 1:N_y
        dudx(end,j) = ( 3*u(end,j) - 4*u(end-1,j) + u(end-2,j) )/(2*dx);
    end
end