function u = shuffle_steps(u)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performs the data swap required prior to advancing to the next time step.
    %
    % This performs the required data transfers for the multistep methods, which store
    % a total of several levels in time. The function assumes we are working with a scalar field,
    % but it can be called on the scalar components of a vector field.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Identify the number of time levels
    num_levels = size(u,3);
    
    % Transfer the time history starting from the oldest available data
    for level = 1:num_levels-1
        u(:,:,level) = u(:,:,level + 1);
    end
end