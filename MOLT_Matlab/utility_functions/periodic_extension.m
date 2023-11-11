function v_ext = periodic_extension(v)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Fills the ghost region of an array "v" using periodic copies.
    %
    % Assumes that v is a 1-D array which has 2 ghost points on each end.
    %
    % Note: v includes the extension, so indices are adjusted accordingly.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    v_ext = v;

    % Left region
    v_ext(1:2) = v(end-4:end-3);

    % Right region
    v_ext(end-1:end) = v(4:5);
    
end