function u = solve_helmholtz_FD6(RHS, alpha, dx, dy)

    [Ny,Nx] = size(RHS);

    assert (Nx == Ny)

    N = Nx*Ny;
    
    I = speye(Nx);

    A = sparse(N);
    % A = zeros(N);

    T = sparse(diag(-4*ones(1,Nx))) + sparse(diag(1*ones(1,Nx-1),1)) + sparse(diag(1*ones(1,Nx-1),-1));

    A(1,mod(1-Nx-1,N)+1) = 1;
    A(1,Nx) = 1;
    A(1,1) = -4;
    A(1,2) = 1;
    A(1,1+Nx) = 1;
    for i = 2:Nx-1
        A(i, mod(i-Nx-1,N)+1) = 1;
        A(i, i-1) = 1;
        A(i, i) = -4;
        A(i, i+1) = 1;
        A(i, i+Nx) = 1;
    end
    A(Nx, 1) = 1;
    A(Nx, Nx-1) = 1;
    A(Nx, Nx) = -4;
    A(Nx, 2*(Nx)) = 1;
    A(Nx, N) = 1;
    
    % A(1*Nx+1:(1+1)*Nx,1*Nx+1:(1+1)*Nx) = T;
    % A((1+1)*Nx,(1+1)*Nx+1 - Nx) = 1;
    % 
    % A(1*Nx+1:(1+1)*Nx,(1+1)*Nx+1:(1+2)*Nx) = I;
    for i = 1:Nx-2
        A(i*Nx+1:(i+1)*Nx,(i-1)*Nx+1:(i)*Nx) = I;
        A(i*Nx+1:(i+1)*Nx,i*Nx+1:(i+1)*Nx) = T;
        A(i*Nx+1:(i+1)*Nx,(i+1)*Nx+1:(i+2)*Nx) = I;
        A((i)*Nx+1,(i+1)*Nx) = 1;
        A((i+1)*Nx,(i+1)*Nx+1 - Nx) = 1;
    end

    A((Nx-1)*Nx+1:(Nx)*Nx,(Nx-2)*Nx+1:(Nx-1)*Nx) = I;

    i = N-Nx+1;
    A(i, i-Nx) = 1;
    A(i, i+Nx-1) = 1;
    A(i, i) = -4;
    A(i, i+1) = 1;
    A(i, mod(i+Nx-1, N) + 1) = 1;
    for i = N-Nx+2:N-1
        A(i, i-Nx) = 1;
        A(i, i-1) = 1;
        A(i, i) = -4;
        A(i, i+1) = 1;
        A(i, mod(i+Nx-1, N) + 1) = 1;
    end

    i = N;

    A(i, i-Nx) = 1;
    A(i, i-1) = 1;
    A(i, i) = -4;
    A(i, Nx*(Nx-1) + 1) = 1;
    A(i, mod(i+Nx-1, N) + 1) = 1;

    alpha2 = alpha*alpha;

    I = speye(N,N);

    H = I - 1/(alpha2*dx*dy)*A;

    r = reshape(RHS',Nx*Ny,1);

    u = H \ r;

    u = reshape(u, Ny, Nx)';

end
