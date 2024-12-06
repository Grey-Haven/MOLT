rng(2);

% Computational domain (normalized wrt L = lam_D)
% To recover the physical units, multiply through by L
a_x = -10*pi/3.0;
b_x =  10*pi/3.0;

N_x = g+1;

dx = (b_x - a_x)/(N_x - 1);

% Grid setup (domain is periodic but we include the last point)
x = a_x:dx:b_x;

% Non-dimensional domain lengths
L_x = b_x - a_x;

kx_deriv_1 = 2*pi/(L_x)*[0:(N_x-1)/2-1, 0, -(N_x-1)/2+1:-1];

kx_deriv_2 = 2*pi/(L_x)*[0:(N_x-1)/2-1, -(N_x)/2, -(N_x-1)/2+1:-1];

% Final time (normalized wrt \omega_{p}^{-1})
T_final = 100.0;

% Number of time steps
N_steps = floor(4e3 / CFL);

dt = T_final/N_steps;

k = 2*pi / L_x;

N_ions = 10000;
N_elec = 20000;

% Normalized mass and charge of the particle species (we suppose there are only 2)
% Sign of the charge is already included in the charge to mass ratio
q_ions = Q_ion/Q;
q_elec = Q_electron/Q;

% Normalized masses
r_ions = M_ion/M;
r_elec = M_electron/M;

% Positions are sampled from a uniform distribution
% and electron velocities come from a 2-D Maxwellian

% Arrays for the particle data
% This includes ions and electrons
x1_ions = zeros(N_ions, 1);
x1_elec = zeros(N_elec, 1);

dx_particles = L_x / N_ions;

%%% Lattice approach for particle intialization
% Don't include the endpoint in the periodic mapping
x1_ions = (a_x:dx_particles:b_x-dx_particles)';
x1_elec(1:N_elec / 2  ) = x1_ions;
x1_elec((N_elec / 2) + 1:end) = x1_ions;

v_b = 1;

% Ions will be stationary for this experiment
v1_ions = zeros(N_ions, 1);

% Electrons have drift velocity in addition to a thermal velocity
v1_elec = zeros(N_elec, 1);

v1_elec(1:N_elec / 2  ) = -v_b;
v1_elec(N_elec / 2 + 1:end) =  v_b;

for i = 1:floor(N_elec / 2)
    x_i = x1_elec(i);
    v1_elec(i) = v1_elec(i) + 5e-4 * sin(k*(x_i - a_x));
    v1_elec(i + floor(N_elec / 2)) = v1_elec(i + floor(N_elec / 2)) + 5e-4 * sin(k*(x_i - a_x));
end

dx_elec = 2 / N_elec;

% Convert velocity to generalized momentum (A = 0 since the total current is zero)
% This is equivalent to the classical momentum
P1_ions = v1_ions*r_ions;
P1_elec = v1_elec*r_elec;

% Compute the normalized particle weights
% L_x and L_y are the non-dimensional domain lengths
w_ions = L_x / N_ions;
w_elec = L_x / N_elec;

N_h = 3;

x1_elec_hist = zeros(N_elec,N_h);

v1_elec_hist = zeros(N_elec,N_h);

P1_elec_hist = zeros(N_elec,N_h);

x1_elec_hist(:,end  ) = x1_elec;
x1_elec_hist(:,end-1) = x1_elec;

v1_elec_hist(:,end  ) = v1_elec;
v1_elec_hist(:,end-1) = v1_elec;
v1_elec_hist(:,end-2) = v1_elec;

P1_elec_hist(:,end  ) = P1_elec;
P1_elec_hist(:,end-1) = P1_elec;

psi = zeros(N_x, N_h);
ddx_psi = zeros(N_x, N_h);
psi_src = zeros(N_x, 1);

A1 = zeros(N_x, N_h);
ddx_A1 = zeros(N_x, N_h);

cell_volumes = dx*ones(N_x, 1);

rho_elec = map_rho_to_mesh_1D(x, dx, x1_elec,  q_elec, cell_volumes, w_elec);
rho_ions = map_rho_to_mesh_1D(x, dx, x1_ions, -q_elec, cell_volumes, w_ions);

rho_elec = enforce_periodicity_1D(rho_elec);
rho_ions = enforce_periodicity_1D(rho_ions);
rho_mesh = rho_elec + rho_ions;

E_hist = zeros(N_steps, 1);

tag = g + "x" + g;

results_path = "./results/" + tag + "/CFL_" + CFL + "/"; % where do we save them?

if ~isfolder(results_path)
   mkdir(results_path)
end

if (enable_plots)
    figure;
    x0=200;
    y0=100;
    width = 1200;
    height = 1200;
    set(gcf,'position',[x0,y0,width,height]);

    vidName = "two_stream_instability" + ".mp4";
    vidObj = VideoWriter(results_path + vidName, 'MPEG-4');
    open(vidObj);
end