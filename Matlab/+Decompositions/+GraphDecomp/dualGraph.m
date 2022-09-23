%% Decompositions.GraphDecomp.dualGraph
% Returns the adjacency matrix and graph Laplacian of the star-expansion
% of the input hypergraph.
%% Syntax
%  [adjMat, lapMat] = lineGraph(HG)
%% Input
% HG - hypergraph object with incidence matrix property obj.IM
%% Output
% * adjMat - adjacency matrix of the star expansion
% * lapMat - graph Laplacian matrix of the starexpansion
%% Disclaimer
%
%% Code
function [A ,L] = dualGraph(HG)
    H = HG.IM;
    [n, e] = size(H);
    A = zeros(n+e);
    for i=1:e
        vxc = find(H(:,i));
        for j=1:length(vxc)
            A(v(j), n+i) = 1;
        end
    end
    A = A + A';
    D = diag(sum(A));
    L = D - A;
end