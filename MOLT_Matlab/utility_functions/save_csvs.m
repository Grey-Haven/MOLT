writeCsvFiles(psi,A1,A2,J1_mesh,J2_mesh,x1_elec_new,x2_elec_new,v1_elec_new,v2_elec_new,x,y,steps,csvPath);

function writeCsvFiles(psi,A1,A2,J1,J2,p_x,p_y,p_vx,p_vy,x,y,s,csv_path)
    
    A1_path = csv_path + "A1\";
    A2_path = csv_path + "A2\";
    J1_path = csv_path + "J1\";
    J2_path = csv_path + "J2\";
    psi_path = csv_path + "psi\";
    part_path = csv_path + "particles\";
    
    A1_title = A1_path + "A1_" + num2str(s) + ".csv";
    A2_title = A2_path + "A2_" + num2str(s) + ".csv";
    J1_title = J1_path + "J1_" + num2str(s) + ".csv";
    J2_title = J2_path + "J2_" + num2str(s) + ".csv";
    psi_title = psi_path + "psi_" + num2str(s) + ".csv";
    part_title = part_path + "particles_" + num2str(s) + ".csv";
    
    if ~isfolder(csv_path)
       mkdir(csv_path)
       mkdir(A1_path);
       mkdir(A2_path);
       mkdir(J1_path);
       mkdir(J2_path);
       mkdir(psi_path);
       mkdir(part_path);
    end

    A1_table = matrix_to_table(x,y,A1,'A1');
    A2_table = matrix_to_table(x,y,A2,'A2');
    J2_table = matrix_to_table(x,y,J2,'J2');
    J1_table = matrix_to_table(x,y,J1,'J1');
    psi_table = matrix_to_table(x,y,psi,'psi');
    part_table = [p_x,p_y,p_vx,p_vy];
    
    writetable(A1_table,A1_title);
    writetable(A2_table,A2_title);
    writetable(J1_table,J1_title);
    writetable(J2_table,J2_title);
    writetable(psi_table,psi_title);
    writematrix(part_table,part_title);
    
end

function theTable = matrix_to_table(x,y,A,col3)
    T = zeros(size(A,1)*size(A,2),3);
    for i = 1:length(x)
        for j = 1:length(y)
            idx = (i-1)*size(A,1) + j;
            T(idx,1) = x(i);
            T(idx,2) = y(j);
            T(idx,3) = A(j,i);
        end
    end
    theTable = array2table(T,'VariableNames',{'x','y',col3});
end