function F_ps = gather_2D_vectorized_multiple_linear(F_meshes, x1_p, x2_p, x, y, dx, dy)
    %%%%%%%%
    % Gathers the field information corresponding to a single particle
    % with coordinates (x1_p, x2_p), but does so with multiple fields to
    % reduce stack traversal.
    %
    % This function uses quadratic splines to map mesh data to the particle
    %
    % This assumes a grid of 
    % [a_x, b_x) x [a_y, b_y), that is, it does not include the right and
    % top boundaries.
    %%%%%%%%

    Nx = length(x)-1;
    Ny = length(y)-1;

    F_ps = zeros(length(x1_p),size(F_meshes,3));

    index_offset = 1;

    lc_x = index_offset+(x1_p - x(1))/dx;
    lc_y = index_offset+(x2_p - y(1))/dy;

    is = floor(lc_x);
    js = floor(lc_y);

    x_node = x(is);
    y_node = y(js);

    fxs = (x1_p - x_node)./dx;
    fys = (x2_p - y_node)./dy;

    isPlusOne = is + 1;
    jsPlusOne = js + 1;

    % The -1 comes from the 1-indexing of Matlab
    isPlusOne = mod(isPlusOne - 1, Nx) + 1;
    jsPlusOne = mod(jsPlusOne - 1, Ny) + 1;

    idx1 = sub2ind(size(F_meshes(:,:,1)),js,is);
    idx2 = sub2ind(size(F_meshes(:,:,1)),js,isPlusOne);
    idx3 = sub2ind(size(F_meshes(:,:,1)),jsPlusOne,is);
    idx4 = sub2ind(size(F_meshes(:,:,1)),jsPlusOne,isPlusOne); 

    for i = 1:size(F_meshes,3)

        F_mesh = F_meshes(:,:,i);

        F1 = F_mesh(idx1).*(1-fys).*(1-fxs);
        F2 = F_mesh(idx2).*(1-fys).*   fxs;
        F3 = F_mesh(idx3).*   fys .*(1-fxs);
        F4 = F_mesh(idx4).*   fys .*   fxs;
        
        F_p = F1+F2+F3+F4;

        F_ps(:,i) = F_p;
    end
end