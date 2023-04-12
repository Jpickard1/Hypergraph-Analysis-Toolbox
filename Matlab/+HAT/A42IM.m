function IM = A42IM(A)
%HYPEREDGE2IM This function returns an incidence matrix corresponding to an
%   adjacency tensor for a 3-uniform hypergraph.
%
% Auth: Joshua Pickard
% Date: April 11, 2022

idxs = find(A ~= 0);
[w,x,y,z] = ind2sub(size(A),idxs);
E = sort([w x y z]')';
E = unique(E, 'rows');
IM = HAT.hyperedges2IM(E);

end

