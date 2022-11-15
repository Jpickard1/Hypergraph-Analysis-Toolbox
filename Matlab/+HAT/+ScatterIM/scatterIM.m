function s = scatterIM(A, ax, color)
%SCATTERIM Summary of this function goes here
%   Detailed explanation goes here
    arguments
        A (:,:) 
        ax = gca
        color = 'k'
    end
    % grab indices of nonzero entries of the incidence matrix to be used in
    % the scatter plot
    [i, j] = find(A);
    s = scatter(ax, j, i, [], color);
end

