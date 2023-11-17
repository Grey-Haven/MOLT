function F_p = gather_2D_vectorized(F_mesh, x1_p, x2_p, x, y, dx, dy)
                        
                        
    index_offset = 1;

    lc_x = index_offset+(x1_p - x(1))/dx;
    lc_y = index_offset+(x2_p - y(1))/dy;

    is = floor(lc_x);
    js = floor(lc_y);

    x_node = x(is)';
    y_node = y(js)';

    fxs = (x1_p - x_node)./dx;
    fys = (x2_p - y_node)./dy;

    isPlusOne = is + 1;
    jsPlusOne = js + 1;

    idx1 = sub2ind(size(F_mesh),js,is);
    idx2 = sub2ind(size(F_mesh),js,isPlusOne);
    idx3 = sub2ind(size(F_mesh),jsPlusOne,is);
    idx4 = sub2ind(size(F_mesh),jsPlusOne,isPlusOne); 

    F1 = F_mesh(idx1).*(1-fys).*(1-fxs);
    F2 = F_mesh(idx2).*fys.*(1-fxs);
    F3 = F_mesh(idx3).*(1-fys).*fxs;
    F4 = F_mesh(idx4).*fys.*fxs;
    
    F_p = F1+F2+F3+F4;

%     F_p = F(i,j,k).*(1-d1).*(1-d2).*(1-d3);  %contribution from (i,j,k)
%     F_p = F_p + F(i+1,j,k).*d1.*(1-d2).*(1-d3);  %(i+1,j,k)
%     F_p = F_p + F(i,j+1,k).*(1-d1).*d2.*(1-d3);  %(i,j+1,k)
%     F_p = F_p + F(i+1,j+1,k).*d1.*d2.*(1-d3);  %(i+1,j+1,k)
% 
%     F_p = F_p + F(i+1,j,k+1).*(1-d1).*d2.*d3;
%     F_p = F_p + F(i+1,j,k+1).*d1.*(1-d2).*d3;  %(i+1,j,k)
%     F_p = F_p + F(i,j+1,k+1).*(1-d1).*d2.*d3;  %(i,j+1,k)
%     F_p = F_p + F(i+1,j+1,k+1).*d1.*d2.*d3;  %(i+1,j+1,k)
%     catch exception
%         throw(exception);
%     end
end