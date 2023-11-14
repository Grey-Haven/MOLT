function dudy = compute_ddy_FD(u, dy)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Computes an y derivative via finite differences
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    N_x = size(u,2);
    N_y = size(u,1);
    
    dudy = zeros(size(u));
    
    % Left boundary (forward diff)
    for i = 1:N_x
        dudy(1,i) = ( -3*u(1,i) + 4*u(2,i) - u(3,i) )/(2*dy);
    end
        
    % Central derivatives
    for i = 1:N_x
        for j = 2:N_y-1
            dudy(j,i) = ( u(j+1,i) - u(j-1,i) )/(2*dy);
        end
    end
            
    % Right boundary (backward diff)
    for i = 1:N_x
        dudy(end,i) = ( 3*u(end,i) - 4*u(end-1,i) + u(end-2,i) )/(2*dy);
    end 
end
