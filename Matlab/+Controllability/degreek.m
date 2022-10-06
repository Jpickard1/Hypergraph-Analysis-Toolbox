function degreeTensor = degreek(edgeSet, numNodes)

% Compute the degree tensor for k-uniform hypergraphs
% edgeSet is a k by k matrix such that each row is a hyperedge
% numNodes is an integer
% Examples: 
%          D = degree3([1 2 3; 2 3 4; 3 4 5], 5)
%          D = degree3([2 3 5; 6 7 8; 5 8 9; 2 6 8], 10)
% by Can Chen, Rahmy Salman
[~, k] = size(edgeSet);
sizeOfDegreeTensor = numNodes * ones(1, k);
degreeTensor = zeros(sizeOfDegreeTensor);

for i = 1:numNodes
    idx = num2cell(i * ones(1, k));
    degreeTensor(idx{:}) = length(find(edgeSet(:)==i));
end
