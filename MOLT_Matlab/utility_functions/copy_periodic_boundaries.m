%---------------------------------------------------------------------
% Assumes an (N-1)x(N-1) domain on an NxN grid
% Copies the first row/column to the Nth row/column
% DIFFERENT THAN ENFORCE PERIODICITY
%---------------------------------------------------------------------
function u = copy_periodic_boundaries(u)
    u(:,end) = u(:,1);
    u(end,:) = u(1,:);
end