function tensorEntropy = tensorEntropy(HG) %edgeSet, numNodes)
%% Tensor Entropy
% Compute the tensor entropy for k-uniform hypergraphs
% edgeSet is a m by k matrix such that each row is a hyperedge
% numNodes is an integer
%
%% Examples: 
%          e = entropyk([1 2 3; 2 3 4; 3 4 5], 5)
%          e = entropyk([2 3 5; 6 7 8; 5 8 9; 2 6 8], 10)
%% Authors
%   Can Chen
%   Rahmy Salman
%   Joshua Pickard

% adjacencyTensor = hypergraphk(edgeSet, numNodes);
% degreeTensor = degreek(edgeSet, numNodes);
% laplacianTensor = degreeTensor-adjacencyTensor;
laplacianTensor = HG.laplacianTensor;
numNodes = size(HG.IM, 1);
order = length(size(laplacianTensor));

laplacianUnfold = reshape(laplacianTensor, numNodes, numNodes^(order-1));

singularValues = svd(laplacianUnfold, 'econ');
normalizedValues = singularValues./sum(singularValues);
normalizedValues = normalizedValues(normalizedValues > 1e-8);
tensorEntropy = -sum(normalizedValues.*log(normalizedValues));

end

