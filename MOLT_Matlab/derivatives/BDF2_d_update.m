% Takes in time history and returns the next step
function u_next = BDF2_d_update(u_hist, u_src, dt)
    u_next = 4/3*u_hist(:,:,end) - 1/3*u_hist(:,:,end-1) - ((2/3)*dt)*u_src;
end