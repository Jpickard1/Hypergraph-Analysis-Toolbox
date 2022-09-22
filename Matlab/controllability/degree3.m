function degreeTensor = degree3(edgeSet, numNodes)

% Compute the degree tensor for 3-uniform hypergraphs
% edgeSet is a m by 3 matrix such that each row is a hyperedge
% numNodes is an integer
% Examples: 
%          D = degree3([1 2 3; 2 3 4; 3 4 5], 5)
%          D = degree3([2 3 5; 6 7 8; 5 8 9; 2 6 8], 10)
% by Can Chen

degreeTensor = zeros(numNodes, numNodes, numNodes);

for i = 1:numNodes
    degreeTensor(i, i, i) = length(find(edgeSet(:)==i));
end
