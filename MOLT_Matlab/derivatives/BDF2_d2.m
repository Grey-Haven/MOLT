function diff = BDF2_d2(u, dt)
    dims = size(u);
    N_h = dims(end);
    num_dimensions = ndims(u);
    otherdims = repmat({':'},1,num_dimensions-1);
    diff = ( u(otherdims{:},N_h) - 8/3*u(otherdims{:},N_h-1) + 22/9*u(otherdims{:},N_h-2) - 8/9*u(otherdims{:},N_h-3) + 1/9*u(otherdims{:},N_h-4) ) / ((2/3)*dt)^2;
end