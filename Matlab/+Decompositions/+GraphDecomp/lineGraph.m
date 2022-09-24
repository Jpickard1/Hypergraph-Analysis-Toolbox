%% Decompositions.GraphDecomp.lineGraph
% Returns the adjacency matrix and graph Laplacian of the line-graph
% of the input hypergraph. Note that the line graph is equivalent to the
% clique-expansion of the dual hypergraph. 
%% Syntax
%  [adjMat, lapMat] = lineGraph(HG)
%% Input
% HG - hypergraph object with incidence matrix property obj.IM
%% Output
% * adjMat - adjacency matrix of the line graph
% * lapMat - graph Laplacian matrix of the line graph
%% Disclaimer
%
%% Code
function [adjMat,lapMat] = lineGraph(HG)
%LINEGRAPH the line graph is the clique expansion of the dual (and the dual
% is the transpose of the original incidence matrix).
% H = HG.IM;

dG = Decompositions.GraphDecomp.dualGraph(HG);
H = Hypergraph('H', dG); %incidenceMatrix(dG)); % Incidence matrix function needs to be changed
[adjMat, lapMat] = Decompositions.GraphDecomp.cliqueGraph(H);
end

