function diff = BDF3_d(u, dt)
    dims = size(u);
    N_h = dims(end);
    num_dimensions = ndims(u);
    otherdims = repmat({':'},1,num_dimensions-1);
    diff = (u(otherdims{:},N_h) - 18/11*u(otherdims{:},N_h-1) + 9/11*u(otherdims{:},N_h-2) - 2/11*u(otherdims{:},N_h-3))/(6/11*dt);
end