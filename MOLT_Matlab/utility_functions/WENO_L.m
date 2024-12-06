function J_L = WENO_L(v,gamma,dx,bdy_ext_indx)
% WENO_L: This code is used to derive the weno interpolation of J_L
% Input: 
%        v: the function f of x
%        bdy_ext_indx: index to show periodic boundary extension
%        dx: mesh size
%        dt: time step size
%        gamma: stability parameter
% Output: the weno interpolation of J_L
gamma = gamma*dx;
epsilon = 1e-6;
len_v = length(v);
J_L = zeros(len_v,1);


cl_11 = ( 6 - 6*gamma + 2*gamma^2 - ( 6 - gamma^2 )*exp(-gamma) )/(6*gamma^3);
cl_12 = -( 6 - 8*gamma + 3*gamma^2 - ( 6 - 2*gamma - 2*gamma^2 )*exp(-gamma) )/(2*gamma^3);
cl_13 = ( 6 - 10*gamma + 6*gamma^2 - ( 6 - 4*gamma - gamma^2 + 2*gamma^3 )*exp(-gamma) )/(2*gamma^3);
cl_14 = -( 6 - 12*gamma + 11*gamma^2 - 6*gamma^3 - ( 6 - 6*gamma + 2*gamma^2)*exp(-gamma) )/(6*gamma^3);

cl_21 = ( 6 - gamma^2 - ( 6 + 6*gamma + 2*gamma^2 )*exp(-gamma) )/(6*gamma^3);
cl_22 = -( 6 - 2*gamma - 2*gamma^2 - ( 6 + 4*gamma - gamma^2 - 2*gamma^3 )*exp(-gamma) )/(2*gamma^3);
cl_23 = ( 6 - 4*gamma - gamma^2 + 2*gamma^3 - ( 6 + 2*gamma - 2*gamma^2 )*exp(-gamma) )/(2*gamma^3);
cl_24 = -( 6 - 6*gamma + 2*gamma^2 - ( 6 - gamma^2 )*exp(-gamma) )/(6*gamma^3);

cl_31 = ( 6 + 6*gamma +2*gamma^2 - ( 6 + 12*gamma + 11*gamma^2 + 6*gamma^3 )*exp(-gamma) )/(6*gamma^3);
cl_32 = -( 6 + 4*gamma - gamma^2 - 2*gamma^3 - ( 6 + 10*gamma + 6*gamma^2 )*exp(-gamma) )/(2*gamma^3 );
cl_33 = ( 6 + 2*gamma - 2*gamma^2 - ( 6 + 8*gamma + 3*gamma^2 )*exp(-gamma) )/(2*gamma^3 );
cl_34 = -( 6 - gamma^2 - ( 6 + 6*gamma + 2*gamma^2 )*exp(-gamma) )/(6*gamma^3);
    
%------------------------------------------------------------------------------------------------------------------
% Compute the linear weights
%------------------------------------------------------------------------------------------------------------------
d1 = ( 60 - 15*gamma^2 + 2*gamma^4 - ( 60 + 60*gamma + 15*gamma^2 - 5*gamma^3 - 3*gamma^4)*exp(-gamma) );
d1 = d1/(10*(gamma^2)*( 6 - 6*gamma + 2*gamma^2 - ( 6 - gamma^2 )*exp(-gamma) ) );

d3 = ( 60 - 60*gamma + 15*gamma^2 + 5*gamma^3 - 3*gamma^4 - ( 60 - 15*gamma^2 + 2*gamma^4)*exp(-gamma) );
d3 = d3/(10*(gamma^2)*( 6 - gamma^2 - ( 6 + 6*gamma + 2*gamma^2 )*exp(-gamma) ) );

d2 = 1 - d1 - d3;


% Check to make sure the left based linear weights are non-negative
if (d1 < 0) || (d2 < 0) || (d3 < 0)
    sprintf("Error: Negative linear weight(s)");
end

bn = 2;
if bdy_ext_indx == "per"
    v2 = zeros(len_v + 2*bn,1);

    v2(1:bn) = v(len_v-2:len_v-1);
    v2(bn+1:len_v+bn) = v';
    v2(len_v+3:len_v+2*bn) = v(2:3);
end

%------------------------------------------------------------------------------------------------------------------
% Compute the local integrals J_{i}^{L} on x_{i-1} to x_{i}, i = 1,...,N+1
%------------------------------------------------------------------------------------------------------------------
for i = 2+bn:len_v+bn

    % Compute the smoothness indicators beta_i
    beta1 = (781)*(-v2(i-3) + 3*v2(i-2) - 3*v2(i-1) +   v2(i))^2 + ...
            (195)*( v2(i-3) - 5*v2(i-2) + 7*v2(i-1) - 3*v2(i))^2 + ...
            (720)*(                       1*v2(i-1) - 1*v2(i))^2;

    beta2 = (781)*(-v2(i-2) + 3*v2(i-1) - 3*v2(i) + v2(i+1))^2 + ...
            (195)*( v2(i-2) -   v2(i-1) -   v2(i) + v2(i+1))^2 + ...
            (720)*(           1*v2(i-1) - 1*v2(i)          )^2;

    beta3 = (781)*(-  v2(i-1) + 3*v2(i) - 3*v2(i+1) + v2(i+2))^2 + ...
            (195)*(-3*v2(i-1) + 7*v2(i) - 5*v2(i+1) + v2(i+2))^2 + ...
            (720)*( 1*v2(i-1) - 1*v2(i)                      )^2;

    % Compute the absolute difference between beta1 and beta3
    % tau = np.abs(beta1 - beta3);

    % % Compute the nonlinear filter
    % beta_max = 1 + ( tau/( epsilon + min(beta1,beta3) ) );
    % beta_min = 1 + ( tau/( epsilon + max(beta1,beta3) ) );
    % 
    % xi_L(i-bn) = beta_min/beta_max; %from 1, 2, ..., N+1 (missing 0th point)

    % Transform the linear weights to nonlinear weights
    omega1 = d1 / (epsilon + beta1)^2;
    omega2 = d2/ (epsilon + beta2)^2;
    omega3 = d3 / (epsilon + beta3)^2;

    % Normalize the weights so they sum to unity
    omega_sum = omega1 + omega2 + omega3;
    omega1 = omega1/omega_sum;
    omega3 = omega3/omega_sum;
    omega2 = 1 - omega1 - omega3;

    % Polynomial interpolants on the substencils
    p1 = cl_11*v2(i-3) + cl_12*v2(i-2) + cl_13*v2(i-1) + cl_14*v2(i  );
    p2 = cl_21*v2(i-2) + cl_22*v2(i-1) + cl_23*v2(i  ) + cl_24*v2(i+1);
    p3 = cl_31*v2(i-1) + cl_32*v2(i  ) + cl_33*v2(i+1) + cl_34*v2(i+2);

    % Compute the integral using the nonlinear weights and the local polynomials
    J_L(i-bn) = omega1*p1 + omega2*p2 + omega3*p3;

end