% Speed of light
c = 2.99792458e08;  % Units of m/s

% Permittivity and permeability of free space
epsilon_0 = 8.854187817e-12; % Units of L^{-3} M^{-1} T^{4} A^{2}
mu_0 = 1.25663706e-06; % Units of MLT^{-2} A^{-2}

% Boltzmann constant in SI units
k_B = 1.38064852e-23; % Units of L^{2} M T^{-2} K^{-1} (energy units)

% Particle species mass parameters
ion_electron_mass_ratio = 10000.0;

electron_charge_mass_ratio = -175882008800.0; % Units of C/kg
ion_charge_mass_ratio = -electron_charge_mass_ratio/ion_electron_mass_ratio; % Units of C/kg

M_electron = (-1.602e-19)/electron_charge_mass_ratio;
M_ion = ion_electron_mass_ratio*M_electron;

Q_electron = electron_charge_mass_ratio*M_electron;
Q_ion = ion_charge_mass_ratio*M_ion;

% Scale for mass [kg]
M = M_electron;

% Scale for (electron) charge [C] (keep this as positive)
Q = 1.602e-19;

% Compute the average macroscopic number density for the plasma [m^{-3}]
n_bar = 10^13; % number density in [m^-3]

% Compute the average macroscopic temperature [K] using lam_D and n_bar
T_bar = 10000; % temperature in Kelvin [K]

% Angular oscillation frequency [rad/s]
w_p = sqrt( ( n_bar*(Q^2) )/( M*epsilon_0 ) );

% Debye length [m]
lam_D = sqrt((epsilon_0 * k_B * T_bar)/(n_bar*Q^2));

% Define the length and time scales from the plasma parameters
L = lam_D; % L has units of [m]
T = 1/w_p; % T has units of [s/rad]

% Compute the thermal velocity V = lam_D * w_p in units of [m/s]
V = L/T;

% Normalized speed of light
kappa = c/V;

% Derived scales for the scalar potential and vector potential
% Be careful: If T is not the plasma period, then we shall have constants in
% front of the wave equations that are not necessarily 1
psi_0 = (M*V^2)/Q;
A_0 = (M*V)/Q;

% Number density of the electrons (same for ions due to quasi-neutrality)
n0 = n_bar; % n0 has units of [m^{-3}]

% Scales used in the Lorentz force
% defined in terms of psi_0 and A_0
E_0 = psi_0/L;
B_0 = A_0/L;

% These are the coefficients on the sources for the wave equations
sigma_1 = (M*epsilon_0)/(n_bar*(Q*T)^2);
sigma_2 = (n_bar*mu_0*(Q*L)^2)/M;

% MOLT stability parameter
% Set for the first-order method
beta_BDF = 1.0;