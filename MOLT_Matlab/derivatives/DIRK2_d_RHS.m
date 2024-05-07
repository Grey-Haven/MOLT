function RHS = DIRK2_d_RHS(rhs_curr, rhs_prev)
    % Qin and Zhang
    b1 = 1/2;
    b2 = 1/2;

    c1 = 1/4;
    c2 = 3/4;
    
    % Pareschi and Russo
    % x = 1/3; 
    % 
    % b1 = 1/2;
    % b2 = 1/2;
    % 
    % c1 = x;
    % c2 = 1 - x;

    RHS_1 = (1-c1)*rhs_prev + c1*rhs_curr;
    RHS_2 = (1-c2)*rhs_prev + c2*rhs_curr;

    RHS = b1*RHS_1 + b2*RHS_2;
end