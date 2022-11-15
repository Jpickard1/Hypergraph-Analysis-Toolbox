function tensorEntropy = entropyk(edgeSet, numNodes)
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

adjacencyTensor = hypergraphk(edgeSet, numNodes);
degreeTensor = degreek(edgeSet, numNodes);
laplacianTensor = degreeTensor-adjacencyTensor;

laplacianUnfold = reshape(laplacianTensor, numNodes, numNodes^2);

singularValues = svd(laplacianUnfold, 'econ');
normalizedValues = singularValues./sum(singularValues);
normalizedValues = normalizedValues(normalizedValues > 0);
tensorEntropy = -sum(normalizedValues.*log(normalizedValues));

end

