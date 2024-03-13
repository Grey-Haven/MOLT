% Takes in time history and returns the next step
% Assumes the last index is the latest
function u_next = BDF3_d_update(u_hist, u_src, dt)
    u_next = 18/11*u_hist(:,:,end) - 9/11*u_hist(:,:,end-1) + 2/11*u_hist(:,:,end-2) - ((6/11)*dt)*u_src;
end