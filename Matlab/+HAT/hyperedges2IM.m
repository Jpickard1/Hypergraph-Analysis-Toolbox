function IM = hyperedges2IM(edgeSet)
%HYPEREDGE2IM This function returns an incidence matrix corresponding to an
%   edge set.
%
% Auth: Joshua Pickard
% Date: November 28, 2022

n = max(max(edgeSet));
e = size(edgeSet,1);
IM = zeros(n, e);
for e=1:size(IM,2)
    IM(edgeSet(e,:), e) = 1;
end

end

