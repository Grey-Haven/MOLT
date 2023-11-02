function dudy = compute_ddy_FD(u, dy)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Computes an y derivative via finite differences
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    N_x = size(u,1);
    N_y = size(u,2);
    
    dudy = zeros(size(u));
    
    % Left boundary (forward diff)
    for i = 1:N_x
        dudy(i,1) = ( -3*u(i,1) + 4*u(i,2) - u(i,3) )/(2*dy);
    end
        
    % Central derivatives
    for i = 1:N_x
        for j = 2:N_y-1
            dudy(i,j) = ( u(i,j+1) - u(i,j-1) )/(2*dy);
        end
    end
            
    % Right boundary (backward diff)
    for i = 1:N_x
        dudy(i,end) = ( 3*u(i,end) - 4*u(i,end-1) + u(i,end-2) )/(2*dy);
    end 
end
