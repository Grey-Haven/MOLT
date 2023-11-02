function u_mesh = map_v_avg_to_mesh(x, y, dx, dy, x1_particles, x2_particles, v1_particles_old, v2_particles_old, v1_particles_new, v2_particles_new)
    u_mesh = zeros(2,length(x),length(y));
    n_dens = zeros(2,length(x),length(y));

    Nx = length(x);
    Ny = length(y);

    for i = 1:length(x1_particles)
        x1_p = x1_particles(i);
        x2_p = x2_particles(i);
        v1_p_old = v1_particles_old(i);
        v2_p_old = v2_particles_old(i);
        v1_p_new = v1_particles_new(i);
        v2_p_new = v2_particles_new(i);
        
        v1_star = 2*v1_p_new - v1_p_old;
        v2_star = 2*v2_p_new - v2_p_old;

        u_mesh(1,:,:) = squeeze(u_mesh(1,:,:)) + scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, v1_star);
        u_mesh(2,:,:) = squeeze(u_mesh(2,:,:)) + scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, v2_star);

        n_dens(1,:,:) = squeeze(n_dens(1,:,:)) + scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, 1);
        n_dens(2,:,:) = squeeze(n_dens(2,:,:)) + scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, 1);
    end

    u_mesh = u_mesh ./ n_dens;
    u_mesh(isnan(u_mesh)) = 0;
end