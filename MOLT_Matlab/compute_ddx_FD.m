function dudx = compute_ddx_FD(u, dx)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Computes an x derivative via finite differences
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    N_x = size(u,2);
    N_y = size(u,1);

    dudx = zeros(size(u));

    % Left boundary (forward diff)
    for j = 1:N_y
        dudx(j,1) = ( -3*u(j,1) + 4*u(j,2) - u(j,3) )/(2*dx);
    end 
    % Central derivatives
    for j = 1:N_y
        for i = 2:N_x-1
            dudx(j,i) = ( u(j,i+1) - u(j,i-1) )/(2*dx);
        end
    end
    % Right boundary (backward diff)
    for j = 1:N_y
        dudx(j,end) = ( 3*u(j,end) - 4*u(j,end-1) + u(j,end-2) )/(2*dx);
    end
end