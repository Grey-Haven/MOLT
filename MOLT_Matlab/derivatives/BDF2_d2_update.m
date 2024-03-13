% Takes in time history and returns the next step
% Assumes the last index is the latest
function u_next = BDF2_d2_update(u_hist, u_src, alpha)
    u_next = 8/3*u_hist(:,:,end) - 22/9*u_hist(:,:,end-1) + 8/9*u_hist(:,:,end-2) - 1/9*u_hist(:,:,end-3) + 1/(alpha^2)*u_src;
end