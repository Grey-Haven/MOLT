function u = enforce_dirichlet(u,a_x,b_x,a_y,b_y)
    u(1,:) = a_x;
    u(end,:) = b_x;
    u(:,1) = a_y;
    u(:,end) = b_y;
end