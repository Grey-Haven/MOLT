close all;
clear;

Nx = 16;
Ny = 16;

dx = 1;
dy = 1;

a_x = -8;
b_x = 8;

a_y = -8;
b_y = 8;

x = a_x:dx:b_x;
y = a_y:dy:b_y;

x2 = a_x:dx/8:b_x;
y2 = a_y:dy/8:b_y;
        
weight = 1;

figure;

onesGrid = ones(Ny,Nx);

x_p1 = 3.5;
y_p1 = 3.5;
x_p2 = 7.75;
y_p2 = 7.75;

% F = scatter_2D_vectorized_cubic(Nx, Ny, [x_p1,x_p2]', [y_p1,y_p2]', x', y', dx, dy, weight);

% surf(x(1:end-1),y(1:end-1),F);

for x_p = x2
    for y_p = y2
        
        F = scatter_2D_vectorized_cubic(Nx, Ny, x_p, y_p, x, y, dx, dy, weight);
        
        surf(x(1:end-1),y(1:end-1),F);
        % zlim([0,1])
        drawnow;
        assert(abs(sum(sum(F)) - 1) < 1e-15);
        
        Foo = gather_2D_vectorized_multiple_cubic(onesGrid, x_p, y_p, x, y, dx, dy);

        assert(abs(Foo - 1) < 1e-15);

    end

end

