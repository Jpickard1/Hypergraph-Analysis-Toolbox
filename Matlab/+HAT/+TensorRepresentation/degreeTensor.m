function D = degreeTensor(HG)
%DEGREETENSOR This function generates the degree tensor of a uniform
% hypergraph.
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: Unknown
%
% Update: Unweighted the degree tensor, March 1, 2023

H = HG.IM;              % Get incidence matrix
eW = HG.edgeWeights;    % Get edge weights
numNodes = size(H, 1);  % Get number of vertices
order = sum(H(:,1), 1); % Get order of hyperedges

% Used to generate linear indices
p = cumprod([1 (numNodes * ones(1, order-1))]);

d = sum(H,2);
D = zeros(numNodes*ones(1,order));
for i=1:numNodes
    LINIDX = ((i*ones(1,order))-1)*p(:)+1;
    D(LINIDX) = d(i);
end

end

