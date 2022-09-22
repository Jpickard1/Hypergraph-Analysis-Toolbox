function tensorEntropy = entropy3(edgeSet, numNodes)

% Compute the tensor entropy for 3-uniform hypergraphs
% edgeSet is a m by 3 matrix such that each row is a hyperedge
% numNodes is an integer
% Examples: 
%          e = entropy3([1 2 3; 2 3 4; 3 4 5], 5)
%          e = entropy3([2 3 5; 6 7 8; 5 8 9; 2 6 8], 10)
% by Can Chen

adjacencyTensor = hypergraph3(edgeSet, numNodes);
degreeTensor = degree3(edgeSet, numNodes);
laplacianTensor = degreeTensor-adjacencyTensor;

laplacianUnfold = reshape(laplacianTensor, numNodes, numNodes^2);

singularValues = svd(laplacianUnfold, 'econ');
normalizedValues = singularValues./sum(singularValues);
normalizedValues = normalizedValues(normalizedValues > 0);
tensorEntropy = -sum(normalizedValues.*log(normalizedValues));




