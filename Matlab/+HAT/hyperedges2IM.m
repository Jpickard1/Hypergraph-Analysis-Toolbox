function IM = hyperedges2IM(edgeSet)
%HYPEREDGE2IM This function returns an incidence matrix corresponding to an
%   edge set.
%
% Auth: Joshua Pickard
% Date: November 28, 2022

n = max(max(edgeSet));
E = size(edgeSet, 1);
IM = zeros(n, E);
for e=1:E
    IM(edgeSet(e,:), e) = 1;
end

end

