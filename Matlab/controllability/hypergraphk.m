function adjacency = hypergraphk(edgeSet, numNodes)

% Compute the adjacency tensor for k-uniform hypergraphs
% edgeSet is a n by k matrix such that each row is a hyperedge
% numNodes is an integer
% Examples: 
%          A = hypergraphk([1 2 3; 2 3 4; 3 4 5], 5)
%          A = hypergraphk([2 3 5; 6 7 8; 5 8 9; 2 6 8], 10)
% by Can Chen, Rahmy Salman
[n, k] = size(edgeSet);
sizeOfAdjacency = numNodes * ones(1, k); % size of each dimension of the adjacency tensor
adjacency = zeros(sizeOfAdjacency);
for i = 1:n
    edgePerms = perms(edgeSet(i, :));%need all permutations of the edge set to create supersymmetry in the adjacency tensor
    for j = 1:length(edgePerms)
        jEdgePerms = num2cell(edgePerms(j, :));%use the permuted edge to index into the adjacency tensor
        adjacency(jEdgePerms{:}) = 1/2;
    end 
end

end

