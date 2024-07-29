function [kinetic_energy, potential_energy, total_energy] = compute_total_energy(phi, A1, A2, x1, x2, P1, P2, x, y, q_s, m_s)    
    % Extract the grid spacings (assumed to be uniform)
    dx = x(2) - x(1);
    dy = y(2) - y(1);
        
    % Interpolate the fields to the particles
    phi_p = gather_2D_vectorized(phi, x1, x2, x, y, dx, dy);
    A1_p = gather_2D_vectorized(A1, x1, x2, x, y, dx, dy);
    A2_p = gather_2D_vectorized(A2, x1, x2, x, y, dx, dy);
    
    % Now we can build the terms that give the energy
    T = (P1 - q_s*A1_p).^2;
    T = T + (P2 - q_s*A2_p).^2;
    T = T * 1./(2*m_s);
    V = q_s*phi_p;
    
    kinetic_energy = sum(T);
    potential_energy = sum(V);
    total_energy = kinetic_energy + potential_energy;
end