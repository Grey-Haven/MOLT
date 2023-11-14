function u_mesh = map_v_avg_to_mesh(x, y, dx, dy, x1_particles, x2_particles, v1_particles_old, v2_particles_old, v1_particles_new, v2_particles_new)
    u_mesh = zeros(length(y),length(x),2);
    u1_mesh = zeros(length(y),length(x));
    u2_mesh = zeros(length(y),length(x));
    n1_dens = zeros(length(y),length(x));
    n2_dens = zeros(length(y),length(x));

    u1_mesh_alt = zeros(length(y),length(x));
    u2_mesh_alt = zeros(length(y),length(x));
    n1_dens_alt = zeros(length(y),length(x));
    n2_dens_alt = zeros(length(y),length(x));

    u1_mesh_vec = zeros(length(y),length(x));
    u2_mesh_vec = zeros(length(y),length(x));
    n1_dens_vec = zeros(length(y),length(x));
    n2_dens_vec = zeros(length(y),length(x));

    Nx = length(x);
    Ny = length(y);

%     for i = 1:length(x1_particles)
%         x1_p = x1_particles(i);
%         x2_p = x2_particles(i);
%         v1_p_old = v1_particles_old(i);
%         v2_p_old = v2_particles_old(i);
%         v1_p_new = v1_particles_new(i);
%         v2_p_new = v2_particles_new(i);
%         
%         v1_star = 2*v1_p_new - v1_p_old;
%         v2_star = 2*v2_p_new - v2_p_old;
% 
%         standard_scatter_1 = scatter_2D(Nx, Ny, x1_p, x2_p, x, y, dx, dy, v1_star);
%         standard_scatter_2 = scatter_2D(Nx, Ny, x1_p, x2_p, x, y, dx, dy, v2_star);
% 
%         alt_scatter_1 = scatter_2D_vectorized(Nx, Ny, x1_p, x2_p, x, y, dx, dy, v1_star);
%         alt_scatter_2 = scatter_2D_vectorized(Nx, Ny, x1_p, x2_p, x, y, dx, dy, v2_star);
% 
%         u1_mesh(:,:) = u1_mesh(:,:) + standard_scatter_1;
%         u2_mesh(:,:) = u2_mesh(:,:) + standard_scatter_2;
% 
%         n1_dens(:,:) = n1_dens(:,:) + scatter_2D(Nx, Ny, x1_p, x2_p, x, y, dx, dy, 1);
%         n2_dens(:,:) = n2_dens(:,:) + scatter_2D(Nx, Ny, x1_p, x2_p, x, y, dx, dy, 1);
% 
%         u1_mesh_alt(:,:) = u1_mesh_alt(:,:) + alt_scatter_1;
%         u2_mesh_alt(:,:) = u2_mesh_alt(:,:) + alt_scatter_2;
% 
%         n1_dens_alt(:,:) = n1_dens_alt(:,:) + scatter_2D_vectorized(Nx, Ny, x1_p, x2_p, x, y, dx, dy, 1);
%         n2_dens_alt(:,:) = n2_dens_alt(:,:) + scatter_2D_vectorized(Nx, Ny, x1_p, x2_p, x, y, dx, dy, 1);
% 
%         assert(norm(norm(alt_scatter_1 - standard_scatter_1)) < 10*eps);
%         assert(norm(norm(alt_scatter_2 - standard_scatter_2)) < 10*eps);
%     end

    v1_star_vec = 2*v1_particles_new - v1_particles_old;
    v2_star_vec = 2*v2_particles_new - v2_particles_old;

    u1_mesh = scatter_2D_vectorized(Nx, Ny, x1_particles, x2_particles, x', y', dx, dy, v1_star_vec);
    u2_mesh = scatter_2D_vectorized(Nx, Ny, x1_particles, x2_particles, x', y', dx, dy, v2_star_vec);

    n1_dens = scatter_2D_vectorized(Nx, Ny, x1_particles, x2_particles, x', y', dx, dy, 1);
    n2_dens = scatter_2D_vectorized(Nx, Ny, x1_particles, x2_particles, x', y', dx, dy, 1);

    u1_mesh = u1_mesh ./ n1_dens;
    u2_mesh = u2_mesh ./ n1_dens;
    u1_mesh(isnan(u1_mesh)) = 0;
    u2_mesh(isnan(u2_mesh)) = 0;

    u_mesh(:,:,1) = u1_mesh;
    u_mesh(:,:,2) = u2_mesh;
end