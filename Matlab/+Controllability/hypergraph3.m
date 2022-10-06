function adjacency = hypergraph3(edgeSet, numNodes)

% Compute the adjacency tensor for 3-uniform hypergraphs
% edgeSet is a m by 3 matrix such that each row is a hyperedge
% numNodes is an integer
% Examples: 
%          A = hypergraph3([1 2 3; 2 3 4; 3 4 5], 5)
%          A = hypergraph3([2 3 5; 6 7 8; 5 8 9; 2 6 8], 10)
% by Can Chen

adjacency = zeros(numNodes, numNodes, numNodes);
for i = 1:size(edgeSet, 1)
    edgePerms = perms(edgeSet(i, :));
    for j = 1:size(edgePerms, 1)
        adjacency(edgePerms(j, 1), edgePerms(j, 2), edgePerms(j, 3)) = 1/2;
    end 
end