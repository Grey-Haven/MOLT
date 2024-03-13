% Takes in time history and returns the next step
% Assumes the last index is the latest
function u_next = BDF3_d2_update(u_hist, u_src, alpha)
    u_next = 396/121*u_hist(:,:,end) - 522/121*u_hist(:,:,end-1) + 368/121*u_hist(:,:,end-2) - 153/121*u_hist(:,:,end-3) + 36/121*u_hist(:,:,end-4) - 4/121*u_hist(:,:,end-5) + 1/(alpha^2)*u_src;
end