function diff = BDF2_d(u, dt)
    dims = size(u);
    N_h = dims(end);
    num_dimensions = ndims(u);
    otherdims = repmat({':'},1,num_dimensions-1);
    diff = ( u(otherdims{:},N_h) - 4/3*u(otherdims{:},N_h-1) + 1/3*u(otherdims{:},N_h-2) ) / ((2/3)*dt);
end