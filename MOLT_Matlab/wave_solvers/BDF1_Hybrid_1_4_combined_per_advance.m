function [u,dudx,dudy] = BDF_Hybrid_1_4_combined_per_advance(u, dudx, dudy, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF,kx_deriv_1,ky_deriv_1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performs the derivative and field advance function for a 2-D scalar field.
    % The function assumes we are working with a scalar field,
    % but it can be called on the scalar components of a vector field.
    %
    % Shuffles for time stepping are performed later, outside of this utility.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    beta_BDF4 = sqrt(15/4);
    
    % dudx1 = BDF1_ddx_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF);
    % 
    % dudy1 = BDF1_ddy_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF);

    % dudx = BDF4_ddx_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF4);
    % 
    % dudy = BDF4_ddy_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF4);
    
    u(:,:,end) = BDF1_advance_per(u, src_data, x, y, t_n, dx, dy, dt, c, beta_BDF);

    u_next = u(1:end-1,1:end-1,end);
    % 
    [N_y,N_x] = size(u(:,:,end));
    Nx = N_x - 1; % Number of nodes in the nonperiodic section of the grid
    Ny = N_y - 1; % Number of nodes in the nonperiodic section of the grid

    u_next_fft_x = fft(u_next,N_x-1,2);
    u_next_fft_y = fft(u_next,N_y-1,1);

    dudx_fft = zeros(N_y,N_x);
    dudy_fft = zeros(N_y,N_x);

    dudx_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*kx_deriv_1 .*u_next_fft_x,N_x-1,2);
    dudy_fft(1:end-1,1:end-1) = ifft(sqrt(-1)*ky_deriv_1'.*u_next_fft_y,N_y-1,1);

    dudx_fft = copy_periodic_boundaries(dudx_fft);
    dudy_fft = copy_periodic_boundaries(dudy_fft);

    dudx = dudx_fft;
    dudy = dudy_fft;

    % dudx_FD4 = zeros(size(dudx));
    % dudy_FD4 = zeros(size(dudy));
    % 
    % for i = 1:Nx
    %     i_idx_m2 = i-2;
    %     i_idx_m1 = i-1;
    %     i_idx    = i+0;
    %     i_idx_p1 = i+1;
    %     i_idx_p2 = i+2;
    %     i_idx_m2 = mod(i_idx_m2 - 1, Nx) + 1;
    %     i_idx_m1 = mod(i_idx_m1 - 1, Nx) + 1;
    %     i_idx    = mod(i_idx    - 1, Nx) + 1;
    %     i_idx_p1 = mod(i_idx_p1 - 1, Nx) + 1;
    %     i_idx_p2 = mod(i_idx_p2 - 1, Nx) + 1;
    %     for j = 1:Ny
    %         j_idx_m2 = j-2;
    %         j_idx_m1 = j-1;
    %         j_idx    = j+0;
    %         j_idx_p1 = j+1;
    %         j_idx_p2 = j+2;
    %         j_idx_m2 = mod(j_idx_m2 - 1, Ny) + 1;
    %         j_idx_m1 = mod(j_idx_m1 - 1, Ny) + 1;
    %         j_idx    = mod(j_idx    - 1, Ny) + 1;
    %         j_idx_p1 = mod(j_idx_p1 - 1, Ny) + 1;
    %         j_idx_p2 = mod(j_idx_p2 - 1, Ny) + 1;
    % 
    %         dudx_FD4(j_idx,i_idx) = (-u_next(j_idx,i_idx_p2) + 8*u_next(j_idx,i_idx_p1) - 8*u_next(j_idx,i_idx_m1) + u_next(j_idx,i_idx_m2)) / (12*dx);
    %         dudy_FD4(j_idx,i_idx) = (-u_next(j_idx_p2,i_idx) + 8*u_next(j_idx_p1,i_idx) - 8*u_next(j_idx_m1,i_idx) + u_next(j_idx_m2,i_idx)) / (12*dy);
    %     end
    % end
    % 
    % dudx_FD4 = copy_periodic_boundaries(dudx_FD4);
    % dudy_FD4 = copy_periodic_boundaries(dudy_FD4);
    % 
    % dudx = dudx_FD4;
    % dudy = dudy_FD4;

    % dudx_FD6 = zeros(size(dudx));
    % dudy_FD6 = zeros(size(dudy));
    % 
    % for i = 1:Nx
    %     i_idx_m3 = i-3;
    %     i_idx_m2 = i-2;
    %     i_idx_m1 = i-1;
    %     i_idx    = i+0;
    %     i_idx_p1 = i+1;
    %     i_idx_p2 = i+2;
    %     i_idx_p3 = i+3;
    %     i_idx_m3 = mod(i_idx_m3 - 1, Nx) + 1;
    %     i_idx_m2 = mod(i_idx_m2 - 1, Nx) + 1;
    %     i_idx_m1 = mod(i_idx_m1 - 1, Nx) + 1;
    %     i_idx    = mod(i_idx    - 1, Nx) + 1;
    %     i_idx_p1 = mod(i_idx_p1 - 1, Nx) + 1;
    %     i_idx_p2 = mod(i_idx_p2 - 1, Nx) + 1;
    %     i_idx_p3 = mod(i_idx_p3 - 1, Nx) + 1;
    %     for j = 1:Ny
    %         j_idx_m3 = j-3;
    %         j_idx_m2 = j-2;
    %         j_idx_m1 = j-1;
    %         j_idx    = j+0;
    %         j_idx_p1 = j+1;
    %         j_idx_p2 = j+2;
    %         j_idx_p3 = j+3;
    %         j_idx_m3 = mod(j_idx_m3 - 1, Nx) + 1;
    %         j_idx_m2 = mod(j_idx_m2 - 1, Ny) + 1;
    %         j_idx_m1 = mod(j_idx_m1 - 1, Ny) + 1;
    %         j_idx    = mod(j_idx    - 1, Ny) + 1;
    %         j_idx_p1 = mod(j_idx_p1 - 1, Ny) + 1;
    %         j_idx_p2 = mod(j_idx_p2 - 1, Ny) + 1;
    %         j_idx_p3 = mod(j_idx_p3 - 1, Nx) + 1;
    % 
    %         dudx_FD6(j_idx,i_idx) = (-1/60*u_next(j_idx,i_idx_m3) + 3/20*u_next(j_idx,i_idx_m2) - 3/4*u_next(j_idx,i_idx_m1) + 3/4*u_next(j_idx,i_idx_p1) - 3/20*u_next(j_idx,i_idx_p2) + 1/60*u_next(j_idx,i_idx_p3)) / (dx);
    %         dudy_FD6(j_idx,i_idx) = (-1/60*u_next(j_idx_m3,i_idx) + 3/20*u_next(j_idx_m2,i_idx) - 3/4*u_next(j_idx_m1,i_idx) + 3/4*u_next(j_idx_p1,i_idx) - 3/20*u_next(j_idx_p2,i_idx) + 1/60*u_next(j_idx_p3,i_idx)) / (dy);
    %     end
    % end
    % 
    % dudx_FD6 = copy_periodic_boundaries(dudx_FD6);
    % dudy_FD6 = copy_periodic_boundaries(dudy_FD6);
    % 
    % dudx = dudx_FD6;
    % dudy = dudy_FD6;

    % dudx_FD8 = zeros(size(dudx));
    % dudy_FD8 = zeros(size(dudy));
    % 
    % for i = 1:Nx
    %     i_idx_m4 = i-4;
    %     i_idx_m3 = i-3;
    %     i_idx_m2 = i-2;
    %     i_idx_m1 = i-1;
    %     i_idx    = i+0;
    %     i_idx_p1 = i+1;
    %     i_idx_p2 = i+2;
    %     i_idx_p3 = i+3;
    %     i_idx_p4 = i+4;
    %     i_idx_m4 = mod(i_idx_m4 - 1, Nx) + 1;
    %     i_idx_m3 = mod(i_idx_m3 - 1, Nx) + 1;
    %     i_idx_m2 = mod(i_idx_m2 - 1, Nx) + 1;
    %     i_idx_m1 = mod(i_idx_m1 - 1, Nx) + 1;
    %     i_idx    = mod(i_idx    - 1, Nx) + 1;
    %     i_idx_p1 = mod(i_idx_p1 - 1, Nx) + 1;
    %     i_idx_p2 = mod(i_idx_p2 - 1, Nx) + 1;
    %     i_idx_p3 = mod(i_idx_p3 - 1, Nx) + 1;
    %     i_idx_p4 = mod(i_idx_p4 - 1, Nx) + 1;
    %     for j = 1:Ny
    %         j_idx_m4 = j-4;
    %         j_idx_m3 = j-3;
    %         j_idx_m2 = j-2;
    %         j_idx_m1 = j-1;
    %         j_idx    = j+0;
    %         j_idx_p1 = j+1;
    %         j_idx_p2 = j+2;
    %         j_idx_p3 = j+3;
    %         j_idx_p4 = j+4;
    %         j_idx_m4 = mod(j_idx_m4 - 1, Nx) + 1;
    %         j_idx_m3 = mod(j_idx_m3 - 1, Nx) + 1;
    %         j_idx_m2 = mod(j_idx_m2 - 1, Ny) + 1;
    %         j_idx_m1 = mod(j_idx_m1 - 1, Ny) + 1;
    %         j_idx    = mod(j_idx    - 1, Ny) + 1;
    %         j_idx_p1 = mod(j_idx_p1 - 1, Ny) + 1;
    %         j_idx_p2 = mod(j_idx_p2 - 1, Ny) + 1;
    %         j_idx_p3 = mod(j_idx_p3 - 1, Nx) + 1;
    %         j_idx_p4 = mod(j_idx_p4 - 1, Nx) + 1;
    % 
    %         dudx_FD8(j_idx,i_idx) = (1/280*u_next(j_idx,i_idx_m4) - 4/105*u_next(j_idx,i_idx_m3) + 1/5*u_next(j_idx,i_idx_m2) - 4/5*u_next(j_idx,i_idx_m1) + 4/5*u_next(j_idx,i_idx_p1) - 1/5*u_next(j_idx,i_idx_p2) + 4/105*u_next(j_idx,i_idx_p3) - 1/280*u_next(j_idx,i_idx_p4)) / (dx);
    %         dudy_FD8(j_idx,i_idx) = (1/280*u_next(j_idx_m4,i_idx) - 4/105*u_next(j_idx_m3,i_idx) + 1/5*u_next(j_idx_m2,i_idx) - 4/5*u_next(j_idx_m1,i_idx) + 4/5*u_next(j_idx_p1,i_idx) - 1/5*u_next(j_idx_p2,i_idx) + 4/105*u_next(j_idx_p3,i_idx) - 1/280*u_next(j_idx_p4,i_idx)) / (dy);
    %     end
    % end
    % 
    % dudx_FD8 = copy_periodic_boundaries(dudx_FD8);
    % dudy_FD8 = copy_periodic_boundaries(dudy_FD8);

    % dudx = dudx_fft;
    % dudy = dudy_fft;

    % subplot(1,5,1);
    % surf(x,y,dudx-dudx1);
    % title("BDF4 - BDF1");
    % subplot(1,5,2);
    % surf(x,y,dudx-dudx_fft);
    % title("BDF4 - BDF-fft");
    % subplot(1,5,3);
    % surf(x,y,dudx-dudx_FD4);
    % title("BDF4 - BDF-FD4");
    % subplot(1,5,4);
    % surf(x,y,dudx-dudx_FD6);
    % title("BDF4 - BDF-FD6");
    % subplot(1,4,5);
    % surf(x,y,dudx-dudx_FD8);
    % title("BDF4 - BDF-FD8");

    % dudx = dudx_FD8;
    % dudy = dudy_FD8;
    
end


function ddx = BDF4_ddx_advance_per(v, src_data, x, y, t, ...
                                    dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates the ddx of the solution to the wave equation using the fourth-order BDF method.
    % By fourth-order, we mean that a fourth-order difference is used for the u_tt term. The splitting
    % error is not addressed by this method. This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Build the convolution integral over the time history R
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);

    % Variables for the integrands
    R = zeros(N_y,N_x); % time history
    tmp = zeros(N_y,N_x); % tmp storage for the inverse

    for i = 1:N_x
        for j = 1:N_y

            % Time history (v doesn't include the extension region)
            R(j,i) = (4/15)*( (77/6)*v(j,i,end-1) - (107/6)*v(j,i,end-2) + 13*v(j,i,end-3) - (61/12)*v(j,i,end-4) + (5/6)*v(j,i,end-5) );

            % Contribution from the source term (at t_{n+1})
            R(j,i) = R(j,i) + ( 1/(alpha^2) )*src_data(j,i);
        end
    end

    % Invert the y operator and apply to R, storing in tmp
    tmp = get_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);
    
    % Invert the x operator and apply to tmp, then store in the derivative array
    ddx = get_ddx_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);
    
    % Shuffle is performed outside this function
end


function ddy = BDF4_ddy_advance_per(v, src_data, x, y, t, ...
                                    dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates the ddy of the solution to the wave equation using the fourth-order BDF method.
    % By fourth-order, we mean that a fourth-order difference is used for the u_tt term. The splitting
    % error is not addressed by this method. This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Build the convolution integral over the time history R
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);
    
    % Variables for the integrands
    R = zeros(N_y, N_x); % time history
    tmp = zeros(N_y, N_x); % tmp storage for the inverse
    
    for i = 1:N_x
        for j = 1:N_y

            % Time history (v doesn't include the extension region)
            R(j,i) = (4/15)*( (77/6)*v(j,i,end-1) - (107/6)*v(j,i,end-2) + 13*v(j,i,end-3) - (61/12)*v(j,i,end-4) + (5/6)*v(j,i,end-5) );

            % Contribution from the source term (at t_{n+1})
            R(j,i) = R(j,i) + ( 1/(alpha^2) )*src_data(j,i);
        end
    end

    % Invert the y operator and apply to R, storing in tmp
    tmp = get_ddy_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);

    % Invert the x operator and apply to tmp, then store in the derivative array
    ddy = get_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);

    % Shuffle is performed outside this function
end

function u = BDF1_advance_per(v, src_data, x, y, t, ...
                              dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates a solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Build the convolution integral over the time history R
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);
    
    % Variables for the integrands
    R   = zeros(N_y, N_x); % time history
    tmp = zeros(N_y, N_x); % tmp storage for the inverse
    
    for i = 1:N_x
        for j = 1:N_y
    
            % Time history (v doesn't include the extension region)
            % There are three time levels here
            R(j,i) = 2*v(j,i,end-1) - v(j,i,end-2);

            % Contribution from the source term (at t_{n+1})
            R(j,i) = R(j,i) + (  1/(alpha^2) )*src_data(j,i);
        end
    end

    % Invert the y operator and apply to R, storing in tmp
    tmp = get_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);

    % Invert the x operator and apply to tmp, then store in the new time level
    u = get_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);

    % Shuffle is performed outside this function
    
end




%%%%%%
% For comparison, delete later
%%%%%%
function ddx = BDF1_ddx_advance_per(v, src_data, x, y, t, ...
                                    dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates the ddx of the solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_data.
    %
    % Source function is implicit in the BDF method.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Build the convolution integral over the time history R
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);
    
    % Variables for the integrands
    R   = zeros(N_y, N_x); % time history
    tmp = zeros(N_y, N_x); % tmp storage for the inverse
    
    for i = 1:N_x
        for j = 1:N_y
    
            % Time history (v doesn't include the extension region)
            % There are three time levels here
            R(j,i) = 2*v(j,i,end-1) - v(j,i,end-2);

            % Contribution from the source term (at t_{n+1})
            R(j,i) = R(j,i) + ( 1/(alpha^2) )*src_data(j,i);
        end
    end
    % Invert the y operator and apply to R, storing in tmp
    tmp = get_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);
    
    % Invert the x operator and apply to tmp, then store in the derivative array
    ddx = get_ddx_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);
    
    % Shuffle is performed outside this function
end

function ddy = BDF1_ddy_advance_per(v, src_data, x, y, t, ...
                                  dx, dy, dt, c, beta)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculates the ddy of the solution to the wave equation using the first-order BDF method. 
    % This function accepts the mesh data v and src_fcn.
    %
    % Note that all arrays are passed by reference, unless otherwise stated.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Build the convolution integral over the time history R
    N_x = length(x);
    N_y = length(y);
    alpha = beta/(c*dt);
    
    % Variables for the integrands
    R   = zeros(N_y, N_x); % time history
    tmp = zeros(N_y, N_x); % tmp storage for the inverse
    
    for i = 1:N_x
        for j = 1:N_y
    
            % Time history (v doesn't include the extension region)
            % There are three time levels here
            R(j,i) = 2*v(j,i,end-1) - v(j,i,end-2);

            % Contribution from the source term (at t_{n+1})
            R(j,i) = R(j,i) + ( 1/(alpha^2) )*src_data(j,i);
        end
    end
            
    % Invert the y operator and apply to R, storing in tmp
    tmp = get_ddy_L_y_inverse_per(R, x, y, dx, dy, dt, c, beta);
    
    % Invert the x operator and apply to tmp, then store in the derivative array
    ddy = get_L_x_inverse_per(tmp, x, y, dx, dy, dt, c, beta);
    
    % Shuffle is performed outside this function
    
end