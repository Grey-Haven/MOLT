rng(2);

% Try a beam with v1 > v2 > 0
% Note that V = c, so we require all
% velocities to be less than 1 in these normalized units

% v_perp = 0.5; % Electron drift velocity in x
% v_parallel = 0.1; % Electrons have uniform velocities in the interval [-v_parallel, v_parallel)
v_perp = kappa / 2;
v_parallel = kappa / 100;

% Base number of particles per species per direction and total
N_px = 500;
N_py = 500;
N_p = N_px*N_py;

assert(mod(N_p, 2) < 1, "Error: The total number of electrons should be even.");

% Normalized mass and charge of the particle species (we suppose there are only 2)
% Sign of the charge is already included in the charge to mass ratio
q_ions = Q_ion/Q;
q_elec = Q_electron/Q;

% Normalized masses
r_ions = M_ion/M;
r_elec = M_electron/M;

% Arrays for the particle data
% This includes ions and electrons
% We double the number of particles to create two groups (but halve the weight)
x1_ions = zeros(2*N_p,1);
x2_ions = zeros(2*N_p,1);

x1_elec = zeros(2*N_p,1);
x2_elec = zeros(2*N_p,1);

% Lattice approach for particle intialization
% Don't include the endpoint in the periodic mapping
dx_p = (b_x-a_x)/N_px;
dy_p = (b_y-a_y)/N_py;
x_p = a_x:dx_p:b_x-dx_p; % np.linspace(a_x, b_x, N_px, endpoint = False)
y_p = a_y:dy_p:b_y-dy_p; % np.linspace(a_y, b_y, N_py, endpoint = False)

for i = 1:N_px
    for j = 1:N_py
        x1_ions((i-1)*N_py+j) = x_p(i);
        x2_ions((i-1)*N_py+j) = y_p(j);
    end
end

% Copy the ion positions for the second group
x1_ions(N_p+1:end) = x1_ions(1:N_p);
x2_ions(N_p+1:end) = x2_ions(1:N_p);

% We'll give ions and electrons identical starting positions
% This will make the problem charge neutral
x1_elec(:) = x1_ions(:);
x2_elec(:) = x2_ions(:);

% Ions are stationary
v1_ions = zeros(size(x1_ions));
v2_ions = zeros(size(x2_ions));

% Electron velocities in x are constant
% We create the return current here to
% ensure that the IC has net zero current
v_perp_velocities = v_perp*ones(N_p,1);
v1_elec = zeros(size(x1_elec));
v1_elec(1:N_p) =  v_perp_velocities;
v1_elec(N_p+1:end) = -v_perp_velocities;

% The velocities in y are uniformly distributed in the interval [-v_parallel, v_parallel)
v_range = 2*v_parallel;
v_parallel_velocities = v_range*rand(N_p,1) - v_range/2;
v2_elec = zeros(size(x2_elec));
v2_elec(1:N_p) =  v_parallel_velocities;
v2_elec(N_p+1:end) = -v_parallel_velocities;

% Convert velocity to generalized momentum (A = 0 since the total current is zero)
v_ions_mag = sqrt(v1_ions.*v1_ions + v2_ions.*v2_ions);
v_elec_mag = sqrt(v1_elec.*v1_elec + v2_elec.*v2_elec);

assert(all(v_ions_mag < kappa) & all(v_elec_mag < kappa), "Velocities exceed the speed of light!\n")

gamma_ions = 1./sqrt(1 - (v_ions_mag/kappa).^2);
gamma_elec = 1./sqrt(1 - (v_elec_mag/kappa).^2);

P1_ions = gamma_ions.*v1_ions.*r_ions;
P2_ions = gamma_ions.*v2_ions.*r_ions;

P1_elec = gamma_elec.*v1_elec.*r_elec;
P2_elec = gamma_elec.*v2_elec.*r_elec;

% Compute the normalized particle weights
% L_x and L_y are the non-dimensional domain lengths
w_ions = (L_x*L_y)/length(x1_ions);
w_elec = (L_x*L_y)/length(x1_elec);

% Domain Parameters:

% Number of grid points to use
N = g+1;
N_x = N;
N_y = N;

tag = g + "x" + g;

disp(tag);

dx = (b_x - a_x)/(N_x - 1);
dy = (b_y - a_y)/(N_y - 1);

% Generate the grid points with the ends included
% Grid is non-dimensional but can put units back with L
x = linspace(a_x, b_x, N_x);
y = linspace(a_y, b_y, N_y);

% dt = 5*dx/kappa
% dt = CFL*dx/(sqrt(2)*kappa);
% N_steps = ceil(T_final/dt);
N_steps = 40000;
dt = T_final / N_steps;

%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN Storage Variables
%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------
% Storage for the integrator
%------------------------------------------------------------------

% Initial position, momentum, and velocity of the particles
% We copy the input data rather than overwrite it
% and we store two time levels of history
%
% We'll assume that the ions remain stationary
% so that we only need to update electrons.


% Make a list for tracking the electron velocity history
% we use this to compute the temperature outside the solver
% This variance is an average of the variance in each direction
v_elec_var_history = [];

% Electron positions
x1_elec_old = x1_elec(:);
x2_elec_old = x2_elec(:);

x1_elec_new = x1_elec(:);
x2_elec_new = x2_elec(:);

% Electron momenta
P1_elec_old = P1_elec(:);
P2_elec_old = P2_elec(:);

P1_elec_new = P1_elec(:);
P2_elec_new = P2_elec(:);

% Electron velocities
v1_elec_old = v1_elec(:);
v2_elec_old = v2_elec(:);

v1_elec_new = v1_elec(:);
v2_elec_new = v2_elec(:);

% Velocity at time t^{n-1} used for the Taylor approx. 
v1_elec_nm1 = v1_elec(:);
v2_elec_nm1 = v2_elec(:);

% Taylor approximated velocity
% v_star = v^{n} + ddt(v^{n})*dt
% which is approximated by
% v^{n} + (v^{n} - v^{n-1})
v1_elec_star = v1_elec(:);
v2_elec_star = v2_elec(:);

% Store the total number of particles for each species
N_ions = length(x1_ions);
N_elec = length(x1_elec_new);

% Mesh/field data
% Need current psi, A1, and A2 and N_h-1 historical steps
% as well as their derivatives
%
% We compute ddt_psi with backwards differences
N_h = 6;

psi = zeros(N_y, N_x, N_h);
ddx_psi = zeros(N_y,N_x);
ddy_psi = zeros(N_y,N_x);
psi_src = zeros(N_y,N_x);
psi_A = zeros(N_y,N_x);
psi_C = zeros(N_y,N_x);

A1 = zeros(N_y, N_x, N_h);
ddx_A1 = zeros(N_y,N_x);
ddy_A1 = zeros(N_y,N_x);
A1_src = zeros(N_y,N_x);

A2 = zeros(N_y, N_x, N_h);
ddx_A2 = zeros(N_y,N_x);
ddy_A2 = zeros(N_y,N_x);
A2_src = zeros(N_y,N_x);

% Other data needed for the evaluation of 
% the gauge and Gauss' law
ddt_psi = zeros(N_y,N_x);
ddt_A1 = zeros(N_y,N_x);
ddt_A2 = zeros(N_y,N_x);

E1 = zeros(N_y,N_x);
E2 = zeros(N_y,N_x);

% Note that from the relation B = curl(A), we identify
% B3 = ddx(A2) - ddy(A1)
B3 = zeros(N_y,N_x);

ddx_E1 = zeros(N_y,N_x);
ddy_E2 = zeros(N_y,N_x);

gauge_residual = zeros(N_y,N_x);
gauss_law_residual = zeros(N_y,N_x);

gauge_error_L2 = zeros(N_steps,1);
gauge_error_inf = zeros(N_steps,1);
gauss_law_error = zeros(N_steps,1);

sum_gauss_law_residual = zeros(N_steps,1);

total_mass = zeros(N_steps,1);
total_energy = zeros(N_steps,1);

% We track two time levels of J (n, n+1)
% Note, we don't need J3 for this model 
% Since ions are stationary J_mesh := J_elec
J_mesh = zeros(N_y,N_x,2); % Idx order: grid indices (y,x), time level

% From https://math.mit.edu/~stevenj/fft-deriv.pdf
% TL;DR 
% For first derivative, assuming N = 2n,
% you want the (N/2)th wavenumber to be zero.
% For the second derivative, you want it to be -2pi/L*(N/2)
% They're the same otherwise.
kx_deriv_1 = 2*pi/(L_x)*[0:(N_x-1)/2-1, 0, -(N_x-1)/2+1:-1];
ky_deriv_1 = 2*pi/(L_y)*[0:(N_y-1)/2-1, 0, -(N_y-1)/2+1:-1];

kx_deriv_2 = 2*pi/(L_x)*[0:(N_x-1)/2-1, -(N_x)/2, -(N_x-1)/2+1:-1];
ky_deriv_2 = 2*pi/(L_y)*[0:(N_y-1)/2-1, -(N_y)/2, -(N_y-1)/2+1:-1];

% More human readable then shifted to matlab's convention:
% dkx = 1 / ((N_x-1) * dx); % Wavenumber increment in x direction
% dky = 1 / ((N_y-1) * dy); % Wavenumber increment in y direction
% kx_deriv_2 = 2*pi*fftshift((-(N_x-1)/2:(N_x-1)/2-1) * dkx);
% ky_deriv_2 = 2*pi*fftshift((-(N_y-1)/2:(N_y-1)/2-1) * dky);

% Compute the cell volumes required in the particle to mesh mapping
% The domain is periodic here, so the first and last cells here are
% identical.
cell_volumes = dx*dy*ones(N_y,N_x);
    
% Current time of the simulation and step counter
t_n = 0.0;

% Ions
rho_ions = map_rho_to_mesh_2D(x, y, dx, dy, ...
                              x1_ions, x2_ions, ...
                              q_ions, cell_volumes, w_ions);

% Electrons
rho_elec = map_rho_to_mesh_2D(x, y, dx, dy, ...
                              x1_elec_new, x2_elec_new, ...
                              q_elec, cell_volumes, w_elec);
% Need to enforce periodicity for the charge on the mesh
rho_ions = enforce_periodicity(rho_ions(:,:));
rho_elec = enforce_periodicity(rho_elec(:,:));

% rho_mesh = rho_ions + rho_elec;
rho_mesh = zeros(size(rho_elec));

% Current
J_mesh = map_J_to_mesh_2D2V(J_mesh(:,:,:), x, y, dx, dy, ...
                        x1_elec_new, x2_elec_new, ...
                        v1_elec_old, v2_elec_old, ...
                        q_elec, cell_volumes, w_elec);

% Need to enforce periodicity for the current on the mesh
J_mesh(:,:,1) = enforce_periodicity(J_mesh(:,:,1));
J_mesh(:,:,2) = enforce_periodicity(J_mesh(:,:,2));

J1_mesh = J_mesh(:,:,1);
J2_mesh = J_mesh(:,:,2);

v_elec_var_history = zeros(N_steps, 1);

rho_hist = zeros(N_steps,1);

Ex_L2_hist = zeros(N_steps,1);
Ey_L2_hist = zeros(N_steps,1);

Bz_L2_hist = zeros(N_steps,1);

% total_mass_ions = get_total_mass_species(rho_ions, cell_volumes, q_ions, r_ions);
% total_mass_elec = get_total_mass_species(rho_elec, cell_volumes, q_elec, r_elec);
% 
% total_energy_ions = get_total_energy(psi(:,:,end), A1(:,:,end), A2(:,:,end), ...
%                                      x1_ions, x2_ions, ...
%                                      P1_ions, P2_ions, ...
%                                      x, y, q_ions, w_ions*r_ions, kappa);
% 
% total_energy_elec = get_total_energy(psi(:,:,end), A1(:,:,end), A2(:,:,end), ...
%                                      x1_elec_new, x2_elec_new, ...
%                                      P1_elec_new, P2_elec_new, ...
%                                      x, y, q_elec, w_elec*r_elec, kappa);
% 
% % Combine the results from both species
% total_mass(1) = total_mass_ions + total_mass_elec;
% total_energy(1) = total_energy_ions + total_energy_elec;

%%%%%%%%%%%%%%%%%%%%%%%
% END Storage Variables
%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN Cold Storage Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tag = (length(x)-1) + "x" + (length(y)-1);
filePath = matlab.desktop.editor.getActiveFilename;
projectRoot = fileparts(filePath);

resultsPath = projectRoot + "/results/conserving/p_mult_" + particle_count_multiplier + ...
              "/CFL_" + CFL + "/" + modification + "/" + update_method_folder + "/" + tag + "/";
figPath = resultsPath + "figures/";
csvPath = resultsPath + "csv_files/";
disp(resultsPath);
create_directories;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END Cold Storage Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN Recording Process Start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if enable_plots
    vidName = "moving_electron_bulk" + ".mp4";
    vidObj = VideoWriter(figPath + vidName, 'MPEG-4');
    open(vidObj);
    
    figure;
    x0=200;
    y0=100;
    width = 1800;
    height = 1200;
    set(gcf,'position',[x0,y0,width,height]);
end

steps = 0;
if (write_csvs)
    save_csvs;
end
if (enable_plots)
    create_plots_weibel;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END Recording Process Start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if debug

    disp(" Numerical reference scalings for this configuration:\n");
    disp(" L (electron skin depth) [m] = " + L);
    disp(" T (angular plasma period) [s/rad] = " + T);
    disp(" V (Characterisitic particle velocity) [m/s] = " + V);
    disp(" n_bar (average number density) [m^{-3}] = " + n_bar);
    disp(" T_bar (average temperature) [K] = " + T_bar);
    
    disp("----------------------------------------------")
    
    disp(" Timestepping information:");
    disp(" N_steps: " + N_steps);
    % if abs(D) > 0
    %     % disp(" alpha (diffision equation):", "{:0.6e}".format(1/np.sqrt(D*dt)));
    % end
    disp(" Approx. CFL (field): " + kappa*dt/dx);

    disp("----------------------------------------------");

    disp(" Dimensional quantities: ");
    disp(" Domain length [m]: " + L*L_x);
    disp(" Plasma period [s]: " + 2*pi*T);
    disp(" Final time [s]: " + 2*pi*T_final*T);
    disp(" dt [s] = " + 2*pi*T*dt);
    disp(" dx [m] = " + L*dx);

    disp("----------------------------------------------");

    disp(" Non-dimensional quantities:");
    disp(" Domain length [non-dimensional]: " + L_x);
    disp(" kappa [non-dimensional] = " + kappa);
    disp(" Final time [non-dimensional]: " + T_final);
    disp(" dt [non-dimensional] = " + dt);
    disp(" dx [non-dimensional] = " + dx);
    disp(" Grid cells per Debye length [non-dimensional]: " + (1/dx));
    disp(" Timesteps per plasma period [non-dimensional]: " + (1/dt));

    % Is the time step small enough?
    assert(dt < dx/6, "Make dt smaller. Use more steps or run to a shorter final time.\n")
end