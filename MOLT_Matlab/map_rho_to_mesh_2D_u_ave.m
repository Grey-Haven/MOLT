function rho_mesh = map_rho_to_mesh_2D_u_ave(x, y, dt, u_mesh, rho_mesh)

    dx = x(2) - x(1);
    dy = y(2) - y(1);

    Nx = length(x);
    Ny = length(y);

    rho_mesh_n = rho_mesh(:,:);
    rho_mesh_k = zeros(size(rho_mesh));

    alpha = max(max(rho_mesh));

    flux = @(w)  alpha*w;
    dflux = @(w) alpha;

    %%%%%%%%%%%%%
    % Find original fluxes
    %%%%%%%%%%%%%

    % for i_idx = 1:Nx-1
    % 
    %     immm = mod(i_idx - 3 - 1, (Nx-1)) + 1;
    %     imm  = mod(i_idx - 2 - 1, (Nx-1)) + 1;
    %     im   = mod(i_idx - 1 - 1, (Nx-1)) + 1;
    %     i    = mod(i_idx - 0 - 1, (Nx-1)) + 1;
    %     ip   = mod(i_idx + 1 - 1, (Nx-1)) + 1;
    %     ipp  = mod(i_idx + 2 - 1, (Nx-1)) + 1;
    %     ippp = mod(i_idx + 3 - 1, (Nx-1)) + 1;
    % 
    %     for j_idx = 1:Ny_1
    % 
    %         jmmm = mod(j_idx - 3 - 1, (Ny-1)) + 1;
    %         jmm  = mod(j_idx - 2 - 1, (Ny-1)) + 1;
    %         jm   = mod(j_idx - 1 - 1, (Ny-1)) + 1;
    %         j    = mod(j_idx - 0 - 1, (Ny-1)) + 1;
    %         jp   = mod(j_idx + 1 - 1, (Ny-1)) + 1;
    %         jpp  = mod(j_idx + 2 - 1, (Ny-1)) + 1;
    %         jppp = mod(j_idx + 3 - 1, (Ny-1)) + 1;
    % 
    %         u_vals_l = [u_mesh(1,immm,j), u_mesh(1,imm,j), u_mesh(1,im,j), u_mesh(1,i,j), u_mesh(1,ip,j), u_mesh(1,ipp,j)];
    %         u_vals_r = [u_mesh(1,imm,j), u_mesh(1,im,j), u_mesh(1,i,j), u_mesh(1,ip,j), u_mesh(1,ipp,j), u_mesh(1,ippp,j)];
    % 
    %         u_vals_d = [u_mesh(2,i,jmmm), u_mesh(2,i,jmm), u_mesh(2,i,jm), u_mesh(2,i,j), u_mesh(2,i,jp), u_mesh(2,i,jpp)];
    %         u_vals_u = [u_mesh(2,i,jmm), u_mesh(2,i,jm), u_mesh(2,i,j), u_mesh(2,i,jp), u_mesh(2,i,jpp), u_mesh(2,i,jppp)];
    % 
    %         rho_vals_l = [rho_mesh(1,immm,j), rho_mesh(1,imm,j), rho_mesh(1,im,j), rho_mesh(1,i,j), rho_mesh(1,ip,j), rho_mesh(1,ipp,j)];
    %         rho_vals_r = [rho_mesh(1,imm,j), rho_mesh(1,im,j), rho_mesh(1,i,j), rho_mesh(1,ip,j), rho_mesh(1,ipp,j), rho_mesh(1,ippp,j)];
    % 
    %         rho_vals_d = [rho_mesh(2,i,jmmm), rho_mesh(2,i,jmm), rho_mesh(2,i,jm), rho_mesh(2,i,j), rho_mesh(2,i,jp), rho_mesh(2,i,jpp)];
    %         rho_vals_u = [rho_mesh(2,i,jmm), rho_mesh(2,i,jm), rho_mesh(2,i,j), rho_mesh(2,i,jp), rho_mesh(2,i,jpp), rho_mesh(2,i,jppp)];
    % 
    %         rho1_flux_n_l = weno_flux_splitting(rho_vals_l.*u_vals_l,flux,dflux);
    %         rho1_flux_n_r = weno_flux_splitting(rho_vals_r.*u_vals_r,flux,dflux);
    % 
    %         rho2_flux_n_d = weno_flux_splitting(rho_vals_d.*u_vals_d,flux,dflux);
    %         rho2_flux_n_u = weno_flux_splitting(rho_vals_u.*u_vals_u,flux,dflux);
    % 
    %         rho_mesh_n(i,j) = rho_mesh_n + dt(1/dx * 1/2*(rho_flux_r - rho_flux_l));
    % 
    %     end
    % end

    %%%%%%%%%%%%%%%%
    % Iterate
    %%%%%%%%%%%%%%%%

    MAX_ITER = 10;
    TOL = 1e-4;

    for k = 1:MAX_ITER

        for i_idx = 1:Nx-1
    
            immm = mod(i_idx - 3 - 1, (Nx-1)) + 1;
            imm  = mod(i_idx - 2 - 1, (Nx-1)) + 1;
            im   = mod(i_idx - 1 - 1, (Nx-1)) + 1;
            i    = mod(i_idx - 0 - 1, (Nx-1)) + 1;
            ip   = mod(i_idx + 1 - 1, (Nx-1)) + 1;
            ipp  = mod(i_idx + 2 - 1, (Nx-1)) + 1;
            ippp = mod(i_idx + 3 - 1, (Nx-1)) + 1;
    
            for j_idx = 1:Ny-1
    
                jmmm = mod(j_idx - 3 - 1, (Ny-1)) + 1;
                jmm  = mod(j_idx - 2 - 1, (Ny-1)) + 1;
                jm   = mod(j_idx - 1 - 1, (Ny-1)) + 1;
                j    = mod(j_idx - 0 - 1, (Ny-1)) + 1;
                jp   = mod(j_idx + 1 - 1, (Ny-1)) + 1;
                jpp  = mod(j_idx + 2 - 1, (Ny-1)) + 1;
                jppp = mod(j_idx + 3 - 1, (Ny-1)) + 1;
    
                %%%
                % Original average velocities arguments
                %%%
    
                u_vals_l = [u_mesh(1,immm,j), u_mesh(1,imm,j), u_mesh(1,im,j), u_mesh(1,i,j), u_mesh(1,ip,j), u_mesh(1,ipp,j)];
                u_vals_r = [u_mesh(1,imm,j), u_mesh(1,im,j), u_mesh(1,i,j), u_mesh(1,ip,j), u_mesh(1,ipp,j), u_mesh(1,ippp,j)];
    
                u_vals_d = [u_mesh(2,i,jmmm), u_mesh(2,i,jmm), u_mesh(2,i,jm), u_mesh(2,i,j), u_mesh(2,i,jp), u_mesh(2,i,jpp)];
                u_vals_u = [u_mesh(2,i,jmm), u_mesh(2,i,jm), u_mesh(2,i,j), u_mesh(2,i,jp), u_mesh(2,i,jpp), u_mesh(2,i,jppp)];
    
                %%%
                % Iterating flux arguments
                %%%
    
                rho_vals_l = [rho_mesh(immm,j), rho_mesh(imm,j), rho_mesh(im,j), rho_mesh(i,j), rho_mesh(ip,j), rho_mesh(ipp,j)];
                rho_vals_r = [rho_mesh(imm,j), rho_mesh(im,j), rho_mesh(i,j), rho_mesh(ip,j), rho_mesh(ipp,j), rho_mesh(ippp,j)];
    
                rho_vals_d = [rho_mesh(i,jmmm), rho_mesh(i,jmm), rho_mesh(i,jm), rho_mesh(i,j), rho_mesh(i,jp), rho_mesh(i,jpp)];
                rho_vals_u = [rho_mesh(i,jmm), rho_mesh(i,jm), rho_mesh(i,j), rho_mesh(i,jp), rho_mesh(i,jpp), rho_mesh(i,jppp)];
    
                %%%
                % Iterating fluxes
                %%%
    
                rho_flux_l = weno_flux_splitting(rho_vals_l.*u_vals_l,flux,dflux);
                rho_flux_r = weno_flux_splitting(rho_vals_r.*u_vals_r,flux,dflux);
    
                rho_flux_d = weno_flux_splitting(rho_vals_d.*u_vals_d,flux,dflux);
                rho_flux_u = weno_flux_splitting(rho_vals_u.*u_vals_u,flux,dflux);
    
                %%%
                % Original flux arguments
                %%%
    
                rho_vals_n_l = [rho_mesh_n(immm,j), rho_mesh_n(imm,j), rho_mesh_n(im,j), rho_mesh_n(i,j), rho_mesh_n(ip,j), rho_mesh_n(ipp,j)];
                rho_vals_n_r = [rho_mesh_n(imm,j), rho_mesh_n(im,j), rho_mesh_n(i,j), rho_mesh_n(ip,j), rho_mesh_n(ipp,j), rho_mesh_n(ippp,j)];
    
                rho_vals_n_d = [rho_mesh_n(i,jmmm), rho_mesh_n(i,jmm), rho_mesh_n(i,jm), rho_mesh_n(i,j), rho_mesh_n(i,jp), rho_mesh_n(i,jpp)];
                rho_vals_n_u = [rho_mesh_n(i,jmm), rho_mesh_n(i,jm), rho_mesh_n(i,j), rho_mesh_n(i,jp), rho_mesh_n(i,jpp), rho_mesh_n(i,jppp)];
    
                %%%
                % Original fluxes
                %%%
    
                rho_flux_n_l = weno_flux_splitting(rho_vals_n_l.*u_vals_l,flux,dflux);
                rho_flux_n_r = weno_flux_splitting(rho_vals_n_r.*u_vals_r,flux,dflux);
    
                rho_flux_n_d = weno_flux_splitting(rho_vals_n_d.*u_vals_d,flux,dflux);
                rho_flux_n_u = weno_flux_splitting(rho_vals_n_u.*u_vals_u,flux,dflux);
    
                rho_mesh_k(i,j) = rho_mesh_n(i,j) + dt*(1/dx * 1/2*(rho_flux_r - rho_flux_l + rho_flux_n_r - rho_flux_n_l) + 1/dy * 1/2*(rho_flux_u - rho_flux_d + rho_flux_n_u - rho_flux_n_d));
    
            end
        end
        % diff = sqrt(sum(sum((rho_mesh - rho_mesh_k).^2)));
        diff = norm(rho_mesh - rho_mesh_k);
        if diff < TOL
            break;
        end
        rho_mesh = rho_mesh_k(:,:);
    end
    disp(k + " " + diff);
end