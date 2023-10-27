function rho_mesh = map_rho_to_mesh_from_J_2D_WENO_Periodic(Nx, Ny, J_mesh, dx, dy, dt)
    %%%%%%%%%%%%%%%%%%
    % Computes the charge density on the mesh using 
    % the standard single level spline maps.
    %
    % Assumes a single species is present
    %%%%%%%%%%%%%%%%%%

    wave_speed = 1;

    flux = @(w)  wave_speed*w;
    dflux = @(w) wave_speed;
    
    sum_J = 0;

    rho_mesh = zeros(Nx,Ny);

    % Scatter particle charge data to the mesh
    for i_idx = 1:Nx

        immm = mod(i_idx - 4, (Nx-1)) + 1;
        imm  = mod(i_idx - 3, (Nx-1)) + 1;
        im   = mod(i_idx - 2, (Nx-1)) + 1;
        i    = mod(i_idx - 1, (Nx-1)) + 1;
        ip   = mod(i_idx + 0, (Nx-1)) + 1;
        ipp  = mod(i_idx + 1, (Nx-1)) + 1;
        ippp = mod(i_idx + 2, (Nx-1)) + 1;

        for j_idx = 1:Ny

            jmmm = mod(j_idx - 4, (Ny-1)) + 1;
            jmm  = mod(j_idx - 3, (Ny-1)) + 1;
            jm   = mod(j_idx - 2, (Ny-1)) + 1;
            j    = mod(j_idx - 1, (Ny-1)) + 1;
            jp   = mod(j_idx + 0, (Ny-1)) + 1;
            jpp  = mod(j_idx + 1, (Ny-1)) + 1;
            jppp  = mod(j_idx + 2, (Ny-1)) + 1;

            J1_vals_l = [J_mesh(1,immm,j), J_mesh(1,imm,j), J_mesh(1,im,j), J_mesh(1,i,j), J_mesh(1,ip,j), J_mesh(1,ipp,j)];
            J1_vals_r = [J_mesh(1,imm,j), J_mesh(1,im,j), J_mesh(1,i,j), J_mesh(1,ip,j), J_mesh(1,ipp,j), J_mesh(1,ippp,j)];

            J2_vals_d = [J_mesh(2,i,jmmm), J_mesh(2,i,jmm), J_mesh(2,i,jm), J_mesh(2,i,j), J_mesh(2,i,jp), J_mesh(2,i,jpp)];
            J2_vals_u = [J_mesh(2,i,jmm), J_mesh(2,i,jm), J_mesh(2,i,j), J_mesh(2,i,jp), J_mesh(2,i,jpp), J_mesh(2,i,jppp)];

            J1_flux_l = weno_flux_splitting(J1_vals_l,flux,dflux);
            J1_flux_r = weno_flux_splitting(J1_vals_r,flux,dflux);

            J2_flux_d = weno_flux_splitting(J2_vals_d,flux,dflux);
            J2_flux_u = weno_flux_splitting(J2_vals_u,flux,dflux);

            % J1_i_plus  = .5*(J_mesh[0,ip,j] + J_mesh[0,i,j])
            % J1_i_minus = .5*(J_mesh[0,i,j] + J_mesh[0,im,j])
            % J2_j_plus  = .5*(J_mesh[1,i,jp] + J_mesh[1,i,j])
            % J2_j_minus = .5*(J_mesh[1,i,j] + J_mesh[1,i,jm])

            % J1_diff = J_mesh[0,ip,j] - J_mesh[0,im,j]
            % J2_diff = J_mesh[1,i,jp] - J_mesh[1,i,jm]

            % print(np.abs(.5*J1_diff - (J1_i_plus - J1_i_minus)))
            % print(np.abs(.5*J2_diff - (J2_j_plus - J2_j_minus)))

            % rho_mesh[i,j] = rho_mesh[i,j] - dt*(1/dx * (J1_i_plus - J1_i_minus) + 1/dy * (J2_j_plus - J2_j_minus))

            rho_mesh(i,j) = rho_mesh(i,j) - dt*(1/dx * (J1_flux_r - J1_flux_l) + 1/dy * (J2_flux_u - J2_flux_d));

            sum_J = sum_J + J1_flux_r - J1_flux_l + J2_flux_u - J2_flux_d;
        end
    end
%     disp(sum_J)

    rho_mesh(:,end) = rho_mesh(:,1);
    rho_mesh(end,:) = rho_mesh(1,:);

end