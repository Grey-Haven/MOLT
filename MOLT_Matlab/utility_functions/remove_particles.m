% preserved_particle_idxs = find(particle_in_zone([x1_elec_new,x2_elec_new], a_x, b_x, a_y, b_y));
% 
% N_p = length(preserved_particle_idxs);
% 
% for i = 1:size(x1_elec_hist,2)
%     x1_elec_hist(1:N_p,i) = x1_elec_hist(preserved_particle_idxs,i);
%     x2_elec_hist(1:N_p,i) = x2_elec_hist(preserved_particle_idxs,i);
%     v1_elec_hist(1:N_p,i) = v1_elec_hist(preserved_particle_idxs,i);
%     v2_elec_hist(1:N_p,i) = v2_elec_hist(preserved_particle_idxs,i);
%     P1_elec_hist(1:N_p,i) = P1_elec_hist(preserved_particle_idxs,i);
%     P2_elec_hist(1:N_p,i) = P2_elec_hist(preserved_particle_idxs,i);
% end
% 
% x1_elec_old(1:N_p) = x1_elec_hist(preserved_particle_idxs);
% x2_elec_old(1:N_p) = x2_elec_old(preserved_particle_idxs);
% 
% x1_elec_new = x1_elec_new(preserved_particle_idxs);
% x2_elec_new = x2_elec_new(preserved_particle_idxs);
% 
% v1_elec_nm1(1:N_p) = v1_elec_nm1(preserved_particle_idxs);
% v2_elec_nm1(1:N_p) = v2_elec_nm1(preserved_particle_idxs);
% 
% v1_elec_old(1:N_p) = v1_elec_old(preserved_particle_idxs);
% v2_elec_old(1:N_p) = v2_elec_old(preserved_particle_idxs);
% 
% P1_elec_old(1:N_p) = P1_elec_old(preserved_particle_idxs);
% P2_elec_old(1:N_p) = P2_elec_old(preserved_particle_idxs);

preserved_particle_idxs = find(particle_in_zone([x1_elec_new,x2_elec_new], a_x, b_x, a_y, b_y));

N_p = length(preserved_particle_idxs);

x1_elec_hist_new = zeros(N_p,size(x1_elec_hist,2));
x2_elec_hist_new = zeros(N_p,size(x2_elec_hist,2));
v1_elec_hist_new = zeros(N_p,size(v1_elec_hist,2));
v2_elec_hist_new = zeros(N_p,size(v2_elec_hist,2));
P1_elec_hist_new = zeros(N_p,size(P1_elec_hist,2));
P2_elec_hist_new = zeros(N_p,size(P2_elec_hist,2));

for i = 1:size(x1_elec_hist,2)
    x1_elec_hist_new(:,i) = x1_elec_hist(preserved_particle_idxs,i);
    x2_elec_hist_new(:,i) = x2_elec_hist(preserved_particle_idxs,i);
    v1_elec_hist_new(:,i) = v1_elec_hist(preserved_particle_idxs,i);
    v2_elec_hist_new(:,i) = v2_elec_hist(preserved_particle_idxs,i);
    P1_elec_hist_new(:,i) = P1_elec_hist(preserved_particle_idxs,i);
    P2_elec_hist_new(:,i) = P2_elec_hist(preserved_particle_idxs,i);
end

x1_elec_hist = x1_elec_hist_new;
x2_elec_hist = x2_elec_hist_new;
v1_elec_hist = v1_elec_hist_new;
v2_elec_hist = v2_elec_hist_new;
P1_elec_hist = P1_elec_hist_new;
P2_elec_hist = P2_elec_hist_new;

x1_elec_old = x1_elec_hist(:,end-1);
x2_elec_old = x2_elec_hist(:,end-1);

x1_elec_new = x1_elec_new(preserved_particle_idxs);
x2_elec_new = x2_elec_new(preserved_particle_idxs);

v1_elec_nm1 = v1_elec_hist(:,end-2);
v2_elec_nm1 = v2_elec_hist(:,end-2);

v1_elec_old = v1_elec_hist(:,end-1);
v2_elec_old = v2_elec_hist(:,end-1);

P1_elec_old = P1_elec_hist(:,end-1);
P2_elec_old = P2_elec_hist(:,end-1);