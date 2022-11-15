%% Decompositions.GraphDecomp.starGraph
%   Returns the adjacency matrix and graph Laplacian of the star-expansion
%   of the input hypergraph.
%% Syntax
%   A = starGraph(HG)
%% Input
% * HG - hypergraph object with incidence matrix property obj.IM
%% Output
% * adjMat - adjacency matrix of the star expansion
%% Disclaimer
%
%% Code
function [adjMat] = starGraph(HG)

H = HG.IM;
[n, m] = size(H);

adjMat = zeros(n + m);
adjMat(n+1:n+m,1:n) = H';
adjMat(1:n,m:m+n) = H;
end

