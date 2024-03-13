% Takes in time history and returns the next step
% Assumes the last index is the latest
function u_next = BDF1_d2_update(u_hist, u_src, alpha)
    u_next = 2*u_hist(:,:,end) - u_hist(:,:,end-1) + 1/(alpha^2)*u_src;
end