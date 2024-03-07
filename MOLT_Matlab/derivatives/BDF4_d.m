function diff = BDF4_d(u, dt)
    dims = size(u);
    N_h = dims(end);
    num_dimensions = ndims(u);
    otherdims = repmat({':'},1,num_dimensions-1);
    diff = (u(otherdims{:},N_h) - 48/25*u(otherdims{:},N_h-1) + 36/25*u(otherdims{:},N_h-2) - 16/25*u(otherdims{:},N_h-3) + 3/25*u(otherdims{:},N_h-4)) / ((12/25)*dt);
end