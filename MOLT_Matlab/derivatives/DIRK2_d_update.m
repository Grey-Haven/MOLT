function u_next = DIRK2_d_update(u_curr, RHS, dt)
    u_next = u_curr + dt*DIRK2_d_RHS(RHS(:,:,end),RHS(:,:,end-1));
end