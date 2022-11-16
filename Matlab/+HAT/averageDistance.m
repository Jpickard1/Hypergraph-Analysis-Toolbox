%% Computations.averageDistance
% Computes the mean pair-wise path distance between nodes in the
% hypergraph. 
%% Syntax
%   d = averageDistance(HG)
%   [d, dmax] = averageDistance(HG)
%% Input 
% * HG - Hypergraph object. HG must represent a k-uniform hypergraph.
%% Output
% * d - Average distance between two nodes in the hypergraph.
% * dmax - Maximum distance between two nodes in the hypergraph.
%% Disclaimer
% The formula for average distance was obtained from equation 30 of the
% paper below.
% 
% Amit Surana, Can Chen, and Indika Rajapakse. "Hypergraph dissimilarity measures." arXiv preprint arXiv:2106.08206 (2021).
%% Code
function [d, dmax] = averageDistance(HG)

A = HG.cliqueGraph;
G = graph(A);
D = distances(G);

n = size(HG.IM,1);

dmax=max(D(:));
d = sum(D, 'all')/2;
d = d/(n*(n-1)/2);

end

