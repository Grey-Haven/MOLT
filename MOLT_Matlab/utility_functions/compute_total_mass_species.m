function total_mass = compute_total_mass_species(rho_s, cell_volumes, q_s, m_s)

    % Total charge is the charge density in each cell times the cell volume, which
    % followed by a sum over the cells
    total_charge = sum(sum(rho_s*cell_volumes));
    
    % Total mass is the total charge divided by charge per particle times the mass of each particle    
    total_mass = m_s*(total_charge/q_s);
end