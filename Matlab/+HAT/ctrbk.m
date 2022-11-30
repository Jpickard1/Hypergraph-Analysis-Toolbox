function ctrbMatrix = ctrbk(HG, inputNodesVector)

% Compute the reduced controllability matrix for k-uniform hypergraphs
% edgeSet is a m by k matrix such that each row is a hyperedge
% numNodes is an integer
% inputNodesVector is a vector containing the nodes are controlled
% Examples: 
%          C = ctrbk([1 2 3; 2 3 4; 3 4 5], 5, [1 2])
%          C = ctrbk([1 2 3 4; 2 3 4 5; 3 4 5 6; 1 4 5 6;], 6, [1 2 3])
% by Can Chen, Rahmy Salman

adjacencyTensor = HG.adjTensor;
numNodes = size(adjacencyTensor, 1);
k = length(size(adjacencyTensor));
adjacencyUnfold = reshape(adjacencyTensor, numNodes, numNodes^(k-1));

ctrbMatrix = HAT.getBmatrix(inputNodesVector, numNodes);

j = 0;
while rank(ctrbMatrix) < numNodes && j < numNodes
    kprod = kronExponentiation(ctrbMatrix, k-1);
    nextCtrbMatrix = adjacencyUnfold*kprod;
    ctrbMatrix = [ctrbMatrix, nextCtrbMatrix]; %#ok<AGROW>
    rankCtrbMatrix = rank(ctrbMatrix);
    [U, ~, ~] = svd(ctrbMatrix, 'econ');
    ctrbMatrix = U(:, 1:rankCtrbMatrix);
    j = j+1;
end 

end

function mat = kronExponentiation(mat1, x)
    mat = mat1;
    for i = 1:x-1
        mat = kron(mat, mat1);
    end
end