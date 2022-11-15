function ctrbMatrix = ctrb3(edgeSet, numNodes, inputNodesVector)

% Compute the reduced controllability matrix for 3-uniform hypergraphs
% edgeSet is a m by 3 matrix such that each row is a hyperedge
% numNodes is an integer
% inputNodesVector is a vector containing the nodes are controlled
% Examples: 
%          C = ctrb3([1 2 3; 2 3 4; 3 4 5], 5, [1 2])
%          C = ctrb3([2 3 5; 6 7 8; 5 8 9; 2 6 8], 10, [3 4 5 6])
% by Can Chen

adjacencyTensor = hypergraph3(edgeSet, numNodes);
adjacencyUnfold = reshape(adjacencyTensor, numNodes, numNodes^2);

ctrbMatrix = getBmatrix(inputNodesVector, numNodes);

j = 0;
while rank(ctrbMatrix) < numNodes && j < numNodes
    nextCtrbMatrix = adjacencyUnfold*kron(ctrbMatrix, ctrbMatrix);
    ctrbMatrix = [ctrbMatrix, nextCtrbMatrix]; %#ok<AGROW>
    rankCtrbMatrix = rank(ctrbMatrix);
    [U, ~, ~] = svd(ctrbMatrix, 'econ');
    ctrbMatrix = U(:, 1:rankCtrbMatrix);
    j = j+1;
end 

end