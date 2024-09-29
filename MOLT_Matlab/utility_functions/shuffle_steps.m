function u = shuffle_steps(u)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Performs the data swap required prior to advancing to the next time step.
    %
    % This performs the required data transfers for the multistep methods, which store
    % a total of several levels in time. The function assumes we are working with a scalar field,
    % but it can be called on the scalar components of a vector field.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Identify the number of time levels
    % num_dimensions = ndims(u);
    % num_levels = size(u,num_dimensions);
    % 
    % otherdims = repmat({':'},1,num_dimensions-1);
    % 
    % % Transfer the time history starting from the oldest available data
    % for level = 1:num_levels-1
    %     u(otherdims{:},level) = u(otherdims{:},level + 1);
    % end
    if (ndims(u) == 2)
        u(:,1:end-1) = u(:,2:end);
    elseif (ndims(u) == 3)
        u(:,:,1:end-1) = u(:,:,2:end);
    else
        ME = MException('ShuffleException','Cannot shuffle an array with more than 2 spatial components');
        throw(ME);
    end
end