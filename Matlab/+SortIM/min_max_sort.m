function An = min_max_sort(A)
%MIN_MAX_SORT Returns a new incidence matrix whose edges are sorted first
%on minimum node index, then on maximum node index.

    Bn = min_max(A);
    [~, idx] = sortrows(Bn, [1, 2]); 
    An = A(:, idx);

    % TODO: learn how sortrows breaks ties!
    % As implemented, min_max_sort considers ONLY the minimum and maximum
    % node indices of a hyperedge. I suspect that the figures in Dotson's
    % PoreC paper break ties using intermediate node values.
end

function B = min_max(A)
%MIN_MAX takes an (m,n) incidence matrix and returns an (m,2) array of the
%minimum and maximum node indices of each hyperedge.
% A: (n,m) double. 
    [n, m] = size(A);
    indices = ((1:n) .* ones(m, 1))';

    % "incidence matrix" with node index in place of ones
    Ai = A .* indices;
    
    % min and max should ignore zero values in the "incidence matrix"
    Ai(Ai == 0) = nan;

    [~, I_max] = max(Ai, [], 1, 'omitnan');
    [~, I_min] = min(Ai, [], 1, 'omitnan');
    B = [I_min; I_max]';
    
end