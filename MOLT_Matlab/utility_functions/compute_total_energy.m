function total_energy = compute_total_energy(phi, A1, A2, x1, x2, P1, P2, x, y, q_s, m_s)    
    % Extract the grid spacings (assumed to be uniform)
    dx = x(2) - x(1);
    dy = y(2) - y(1);
        
    % Interpolate the fields to the particles
    phi_p = gather_2D_vectorized(phi, x1, x2, x, y, dx, dy);
    A1_p = gather_2D_vectorized(A1, x1, x2, x, y, dx, dy);
    A2_p = gather_2D_vectorized(A2, x1, x2, x, y, dx, dy);
    
    % Now we can build the terms that give the energy
    tmp = (P1 - q_s*A1_p).^2;
    tmp = tmp + (P2 - q_s*A2_p).^2;
    tmp = tmp * 1./(2*m_s);
    tmp = tmp + tmp + q_s*phi_p;

    total_energy = sum(tmp);
end