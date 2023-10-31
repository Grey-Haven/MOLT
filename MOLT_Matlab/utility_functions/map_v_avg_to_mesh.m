function u_mesh = map_v_avg_to_mesh(x, y, dx, dy, x1_particles, x2_particles, v1_particles, v2_particles)
    u_mesh = zeros(2,length(x),length(y));
    n_dens = zeros(2,length(x),length(y));

    Nx = length(x);
    Ny = length(y);

    for i = 1:length(x1_particles)
        x1_p = x1_particles(i);
        x2_p = x2_particles(i);
        v1_p = v1_particles(i);
        v2_p = v2_particles(i);

        u_mesh(1,:,:) = squeeze(u_mesh(1,:,:)) + scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, v1_p);
        u_mesh(2,:,:) = squeeze(u_mesh(2,:,:)) + scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, v2_p);

        n_dens(1,:,:) = squeeze(n_dens(1,:,:)) + scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, 1);
        n_dens(2,:,:) = squeeze(n_dens(2,:,:)) + scatter_2D_ave_velocity(Nx, Ny, x1_p, x2_p, x, y, dx, dy, 1);
    end

    u_mesh = u_mesh ./ n_dens;
    u_mesh(isnan(u_mesh)) = 0;
end