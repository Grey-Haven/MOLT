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

L_x = 32.0;
L_y = 32.0;

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
dt = dx/(sqrt(2)*kappa);
T_final = 10;
N_steps = int64(T_final/dt);

v_ave_mag = 1;

v1_drift = kappa/100;
v2_drift = 0;

% Number of particles for each species
N_p = int64(2.5e5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END Domain Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN Code Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
debug = true;
save_results = true; % do we save the figures created?
save_csvs = false;
write_stride = 100; % save results every n timesteps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END Code Parameters
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

sig_x = .1*(b_x - a_x);
sig_y = .1*(b_y - a_y);

particle_positions_elec = sig_x*randn(N_p,2) + x_0;
particle_positions_ions = sig_y*randn(N_p,2) + x_0;

x1_elec = particle_positions_elec(:,1);
x2_elec = particle_positions_elec(:,2);

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

    disp("Non-dimensional quantities:\n");
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