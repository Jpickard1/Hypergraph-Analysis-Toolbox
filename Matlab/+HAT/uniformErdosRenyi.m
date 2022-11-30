function HG = uniformErdosRenyi(v, e, k)
%UNIFORMERDOSRENYI This function constructs a k-uniform Erdos-Renyi
%   (random) hypergraph.
% Inputs
% * v: number of vertices
% * e: number of edges
% * k: order of hypergraph
% Auth: Joshua Pickard
%       jpic@umich.edu

IM = zeros(v, e);
for i=1:e
    idx = randsample(v, k);
    IM(idx, i) = 1;
end
HG = Hypergraph('IM', IM);

end

