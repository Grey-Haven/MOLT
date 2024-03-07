function diff = BDF1_d(u, dt)
    dims = size(u);
    N_h = dims(end);
    num_dimensions = ndims(u);
    otherdims = repmat({':'},1,num_dimensions-1);
    diff = ( u(otherdims{:},N_h) - u(otherdims{:},N_h-1) ) / (dt);
end