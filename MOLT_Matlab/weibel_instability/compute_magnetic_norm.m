function norm_B = compute_magnetic_norm(B3, dx, dy)
    norm_B = get_L_2_error(B3, zeros(size(B3)), dx*dy);
end