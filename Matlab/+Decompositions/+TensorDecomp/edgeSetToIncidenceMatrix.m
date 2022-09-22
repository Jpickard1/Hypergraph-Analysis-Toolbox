%% Decompositions.TensorDecomp.edgeSetToIncidenceMatrix.m
% Creates an incidence matrix from an edge set. An edge set is a matrix
% with numEdges rows and sizeOfEdges columns where each row is a hyperedge.
% 
%% Syntax
% 
%% Input
%
%% Output
% 
%% Disclaimer
%
%% Code 
function H = edgeSetToIncidenceMatrix(edgeSet)
    m = size(edgeSet, 1);
    n = max(edgeSet, [], 'all');
    H = zeros(n, m);
    for i = 1:m
        edge = edgeSet(i, :);
        H(edge, i) = 1;
    end
end

