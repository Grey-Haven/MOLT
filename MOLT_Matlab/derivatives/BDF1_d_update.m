% Takes in time history and returns the next step
function u_next = BDF1_d_update(u_hist, u_src, dt)
    u_next = u_hist(:,:,end) + dt*u_src;
end