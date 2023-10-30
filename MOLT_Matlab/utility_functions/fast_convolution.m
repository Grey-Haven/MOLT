function [I_L,I_R] = fast_convolution(I_L, I_R, alpha, dx)

    N = length(I_L);
    
    % Precompute the recursion weight
    weight = exp( -alpha*dx );
    
    % Perform the sweeps to the right
    for i = 2:N
        I_L(i) = weight*I_L(i-1) + I_L(i);
    end
    
    % Perform the sweeps to the left
    for i = N-1:-1:1
        I_R(i) = weight*I_R(i+1) + I_R(i);
    end
end