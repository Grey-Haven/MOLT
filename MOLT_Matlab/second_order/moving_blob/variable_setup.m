rng(2);

% Assumes g (grid_refinement) and mesh independent variables
% have been established
N = g+1;
% Number of grid points to use
N_x = N;
N_y = N;

tag = g + "x" + g;

disp(tag);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN Domain Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L_x = 16.0;
L_y = 16.0;

a_x = -L_x/2;
b_x = L_x/2;

a_y = -L_y/2;
b_y =  L_y/2;

dx = (b_x - a_x)/(N_x - 1);
dy = (b_y - a_y)/(N_y - 1);

% Generate the grid points with the ends included
% Grid is non-dimensional but can put units back with L
x = linspace(a_x, b_x, N_x);
y = linspace(a_y, b_y, N_y);

% dt = 5*dx/kappa
dt = CFL*dx/(sqrt(2)*kappa);
N_steps = ceil(T_final/dt);

v_ave_mag = 1;

v1_drift = kappa/100;
% v1_drift = 0;
v2_drift = kappa/100;
% v2_drift = 0;

% Number of particles for each species
N_p = floor(2.5e4);
% N_p = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END Domain Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN Derived Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% More setup
a_x = x(1);
b_x = x(end);
a_y = y(1);
b_y = y(end);

x1_ions = zeros(N_p,1);
x2_ions = zeros(N_p,1);

x1_elec = zeros(N_p,1);
x2_elec = zeros(N_p,1);

%%% Sampling approach for particle intialization
% % Generate a set of uniform samples in the domain for ions and electrons
xy_min = [a_x, a_y];
xy_max = [b_x, b_y];

% % Create a 2-D array where the columns are x1 and x2 position coordinates
% particle_positions_elec = np.random.uniform(low=xy_min, high=xy_max, size=(N_p,2))
% particle_positions_ions = np.random.uniform(low=xy_min, high=xy_max, size=(N_p,2))

x_0 = (a_x + b_x) / 2;
y_0 = (a_y + b_y) / 2;

x_offset = (b_x - a_x)/4;
y_offset = (b_y - a_y)/4;

sig_x = .05*(b_x - a_x);
sig_y = .05*(b_y - a_y);

particle_positions_elec = sig_x*randn(N_p,2) + [x_0, y_0]; %+ [x_offset, y_offset];
particle_positions_ions = sig_y*randn(N_p,2) + [x_0, y_0];

% x1_elec = particle_positions_elec(:,1);
% x2_elec = particle_positions_elec(:,2);

x1_elec = particle_positions_ions(:,1);
x2_elec = particle_positions_ions(:,2);

x1_ions = particle_positions_ions(:,1);
x2_ions = particle_positions_ions(:,2);

% Normalized masses
r_ions = M_ion/M;
r_elec = M_electron/M;

% Normalized mass and charge of the particle species (we suppose there are only 2)
% Sign of the charge is already included in the charge to mass ratio
q_ions = Q_ion/Q;
q_elec = Q_electron/Q;


% Ions will be stationary for this experiment
v1_ions = zeros(N_p,1);
v2_ions = zeros(N_p,1);

% Sample the electron velocities from a 2-D Maxwellian
% Result is stored as a 2-D array
electron_velocities = randn(N_p,2);

% Electrons have drift velocity in addition to a thermal velocity
v1_elec = v_ave_mag*electron_velocities(:,1) + v1_drift;
v2_elec = v_ave_mag*electron_velocities(:,2) + v2_drift;

% Convert velocity to generalized momentum (A = 0 since the total current is zero)
% This is equivalent to the classical momentum
P1_ions = v1_ions*r_ions;
P2_ions = v2_ions*r_ions;

P1_elec = v1_elec*r_elec;
P2_elec = v2_elec*r_elec;

% Compute the normalized particle weights
% L_x and L_y are the non-dimensional domain lengths
w_ions = (L_x*L_y)/N_p;
w_elec = (L_x*L_y)/N_p;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END Derived Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% Number of steps in the past to store.
N_h = 6;

% Make a list for tracking the electron velocity history
% we use this to compute the temperature outside the solver
% This variance is an average of the variance in each direction
v_elec_var_history = [];

x1_elec_hist = zeros(N_p,N_h);
x2_elec_hist = zeros(N_p,N_h);

v1_elec_hist = zeros(N_p,N_h);
v2_elec_hist = zeros(N_p,N_h);

P1_elec_hist = zeros(N_p,N_h);
P2_elec_hist = zeros(N_p,N_h);

x1_elec_hist(:,end  ) = x1_elec;
x2_elec_hist(:,end  ) = x2_elec;
x1_elec_hist(:,end-1) = x1_elec;
x2_elec_hist(:,end-1) = x2_elec;

v1_elec_hist(:,end  ) = v1_elec;
v2_elec_hist(:,end  ) = v2_elec;
v1_elec_hist(:,end-1) = v1_elec;
v2_elec_hist(:,end-1) = v2_elec;
v1_elec_hist(:,end-2) = v1_elec;
v2_elec_hist(:,end-2) = v2_elec;

P1_elec_hist(:,end  ) = P1_elec;
P2_elec_hist(:,end  ) = P2_elec;
P1_elec_hist(:,end-1) = P1_elec;
P2_elec_hist(:,end-1) = P2_elec;

% Store the total number of particles for each species
N_ions = length(x1_ions);
N_elec = length(x1_elec);

% Mesh/field data
% Need current psi, A1, and A2 and N_h-1 historical steps
% as well as their derivatives
%
% We compute ddt_psi with backwards differences

psi = zeros(N_y, N_x, N_h);
ddx_psi = zeros(N_y, N_x, N_h);
ddy_psi = zeros(N_y, N_x, N_h);
ddt_psi = zeros(N_y,N_x);

psi_src = zeros(N_y, N_x);
psi_src_hist = zeros([N_y,N_x,2]);
psi_A = zeros(N_y, N_x);
psi_C = zeros(N_y, N_x);

A1 = zeros(N_y, N_x, N_h);
ddx_A1 = zeros(N_y, N_x, N_h);
ddy_A1 = zeros(N_y, N_x, N_h);

A1_src = zeros(N_y, N_x);
A1_src_hist = zeros([N_y,N_x,2]);

A2 = zeros(N_y, N_x, N_h);
ddx_A2 = zeros(N_y, N_x, N_h);
ddy_A2 = zeros(N_y, N_x, N_h);

A2_src = zeros(N_y, N_x);
A2_src_hist = zeros([N_y,N_x,2]);

% Other data needed for the evaluation of 
% the gauge and Gauss' law
ddt_psi_hist = zeros(N_y,N_x,N_h);
ddt_A1_hist = zeros(N_y,N_x,N_h);
ddt_A2_hist = zeros(N_y,N_x,N_h);

div_A_prev = zeros(N_y,N_x);
div_A_curr = zeros(N_y,N_x);

E1 = zeros(N_y,N_x);
E2 = zeros(N_y,N_x);

% Note that from the relation B = curl(A), we identify
% B3 = ddx(A2) - ddy(A1)
B3 = zeros(N_y,N_x);

ddx_E1 = zeros(N_y,N_x);
ddy_E2 = zeros(N_y,N_x);

gauge_residual = zeros(N_y,N_x);
gauss_residual = zeros(N_y,N_x);

gauss_law_potential_res = zeros(N_y,N_x);
gauss_law_gauge_res = zeros(N_y,N_x);
gauss_law_field_res = zeros(N_y,N_x);

gauge_error_L2 = zeros(N_steps,1);
gauge_error_inf = zeros(N_steps,1);
gauss_law_error = zeros(N_steps,1);
sum_gauss_law_residual = zeros(N_steps,1);

gauss_law_potential_err_L2 = zeros(N_steps,1);
gauss_law_potential_err_inf = zeros(N_steps,1);
gauss_law_gauge_err_L2 = zeros(N_steps,1);
gauss_law_gauge_err_inf = zeros(N_steps,1);
gauss_law_field_err_L2 = zeros(N_steps,1);
gauss_law_field_err_inf = zeros(N_steps,1);

% Note, we don't need J3 for this model 
% Since ions are stationary J_mesh := J_elec
% J_mesh = zeros(N_y,N_x,2); % Idx order: grid indices (y,x), {1,2} = {Jx,Jy}
J1_mesh = zeros(N_y,N_x,N_h);
J2_mesh = zeros(N_y,N_x,N_h);

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
                              x1_elec_hist(:,end), x2_elec_hist(:,end), ...
                              q_elec, cell_volumes, w_elec);
% Need to enforce periodicity for the charge on the mesh
rho_ions = enforce_periodicity(rho_ions(:,:));
rho_elec = enforce_periodicity(rho_elec(:,:));

% rho_mesh = rho_ions + rho_elec;
rho_mesh = zeros([size(rho_elec),N_h]);

% Current
J_mesh = map_J_to_mesh_2D2V(x, y, dx, dy, ...
                            x1_elec_hist(:,end), x2_elec_hist(:,end), ...
                            v1_elec_hist(:,end), v1_elec_hist(:,end), ...
                            q_elec, cell_volumes, w_elec);

% Need to enforce periodicity for the current on the mesh
J_mesh(:,:,1) = enforce_periodicity(J_mesh(:,:,1));
J_mesh(:,:,2) = enforce_periodicity(J_mesh(:,:,2));

J1_mesh(:,:,end) = J_mesh(:,:,1);
J2_mesh(:,:,end) = J_mesh(:,:,2);

% Initial guesses for the previous two timesteps (-1/2 and -3/2)
% J1_mesh(:,:,end-1) = J_mesh(:,:,1);
% J2_mesh(:,:,end-1) = J_mesh(:,:,2);
% 
% J1_mesh(:,:,end-2) = J_mesh(:,:,1);
% J2_mesh(:,:,end-2) = J_mesh(:,:,2);

v_elec_var_history = zeros(N_steps, 1);

rho_hist = zeros(N_steps,1);

%%%%%%%%%%%%%%%%%%%%%%%
% END Storage Variables
%%%%%%%%%%%%%%%%%%%%%%%

if debug

    disp("Numerical reference scalings for this configuration:");
    disp(" L (Max domain length) [m] = " + L);
    disp(" T (particle crossing time) [s] = " + T);
    disp(" V (beam injection velocity) [m/s] = " + V);
    disp(" n_bar (average number density) [m^{-3}] = " + n_bar);

    disp("----------------------------------------------")

    disp("Timestepping information:");
    disp(" N_steps: " + N_steps);
    disp(" Field CFL: " + kappa*dt/min(dx,dy));
    disp(" Particle CFL: " + v_ave_mag*dt/min(dx,dy));
        
    disp("----------------------------------------------")

    disp("Dimensional quantities:")
    disp(" Domain length in x [m]: " + L*L_x); % L_x is non-dimensional
    disp(" Domain length in y [m]: " + L*L_y); % L_y is non-dimensional
    disp(" Final time [s]: " + T_final*T);

    % dt and dx are both non-dimensional
    disp(" dx [m] = " + L*dx);
    disp(" dy [m] = " + L*dy);
    disp(" dt [s] = " + T*dt);

    disp("----------------------------------------------")

    disp("Non-dimensional quantities:");
    disp(" Domain length in x [non-dimensional]: " + L_x);
    disp(" Domain length in y [non-dimensional]: " + L_y);
    disp(" v_ave_mag/c: " + V*v_ave_mag/c); % v_injection is scaled by V
    disp(" kappa [non-dimensional] = " + kappa);
    disp(" Final time [non-dimensional]: " + T_final);
    disp(" sigma_1 [non-dimensional] = " + sigma_1);
    disp(" sigma_2 [non-dimensional] = " + sigma_2);
    disp(" dx [non-dimensional] = " + dx);
    disp(" dy [non-dimensional] = " + dy);
    disp(" dt [non-dimensional] = " + dt);

    % Is the time step small enough?
    assert(dt < dx/6, "Make dt smaller. Use more steps or run to a shorter final time.\n")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN Cold Storage Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tag = (length(x)-1) + "x" + (length(y)-1);
filePath = matlab.desktop.editor.getActiveFilename;
projectRoot = fileparts(filePath);

resultsPath = projectRoot + "/results/" + gauge_correction + "/linear_interpolation/" + update_method_folder + "/" + tag + "/";
figPath = resultsPath + "figures/";
csvPath = resultsPath + "csv_files/";
disp(resultsPath);
create_directories;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END Cold Storage Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (enable_plots)
    figure;
    x0=200;
    y0=100;
    width = 1800;
    height = 1200;
    set(gcf,'position',[x0,y0,width,height]);

    vidName = "moving_electron_bulk" + ".mp4";
    vidObj = VideoWriter(figPath + vidName, 'MPEG-4');
    open(vidObj);
end