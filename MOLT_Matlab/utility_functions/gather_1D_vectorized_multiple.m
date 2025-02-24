function F_ps = gather_1D_vectorized_multiple(F_meshes, x1_p, x, dx)
                        
    F_ps = zeros(length(x1_p),size(F_meshes,2));

    index_offset = 1;

    lc_x = index_offset+(x1_p - x(1))/dx;

    is = floor(lc_x);

    x_node = x(is)';

    fxs = (x1_p - x_node)./dx;

    isPlusOne = is + 1;

    idx1 = is;
    idx2 = isPlusOne;

    for i = 1:size(F_meshes,2)

        F_mesh = F_meshes(:,i);

        F1 = F_mesh(idx1).*(1-fxs);
        F2 = F_mesh(idx2).*   fxs;
        
        F_p = F1+F2;

        F_ps(:,i) = F_p;
    end
end