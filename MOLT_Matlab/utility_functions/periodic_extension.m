function [] = periodic_extension(v)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Fills the ghost region of an array "v" using periodic copies.
    %
    % Assumes that v is a 1-D array which has 2 ghost points on each end.
    %
    % Note: v includes the extension, so indices are adjusted accordingly.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Left region
    v(1:3) = v(end-5:end-3);

    % Right region
    v(end-2:end) = v(4:6);
    
end