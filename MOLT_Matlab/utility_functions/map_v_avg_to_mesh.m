function u_mesh = map_v_avg_to_mesh(x, y, dx, dy, x1_particles, x2_particles, v1_particles_old, v2_particles_old, v1_particles_new, v2_particles_new)

    u_mesh = zeros(length(y),length(x),2);

    Nx = length(x);
    Ny = length(y);

    v1_star_vec = 2*v1_particles_new - v1_particles_old;
    v2_star_vec = 2*v2_particles_new - v2_particles_old;

    u1_mesh = scatter_2D_vectorized(Nx, Ny, x1_particles, x2_particles, x', y', dx, dy, v1_star_vec);
    u2_mesh = scatter_2D_vectorized(Nx, Ny, x1_particles, x2_particles, x', y', dx, dy, v2_star_vec);

    % n_dens  = scatter_2D_vectorized(Nx, Ny, x1_particles, x2_particles, x', y', dx, dy, 1);

    % u1_mesh = u1_mesh ./ n_dens;
    % u2_mesh = u2_mesh ./ n_dens;
    % u1_mesh(isnan(u1_mesh)) = 0;
    % u2_mesh(isnan(u2_mesh)) = 0;

    u_mesh(:,:,1) = u1_mesh;
    u_mesh(:,:,2) = u2_mesh;
end