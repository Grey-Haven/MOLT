function x_shift = periodic_shift(x, a, L)
    %%%%%%%%%%%%%%%    
    % Performs an element-wise mod by the domain length "L"
    % along a coordinate axis whose left-most point is x = a.
    %%%%%%%%%%%%%%%
    x_shift = x - L*floor( (x - a)/L);
end