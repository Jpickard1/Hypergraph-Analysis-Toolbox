%% Decompositions.GraphDecomp.RodriguezLaplacian
% Returns the adjacency matrix and laplacian matrix of a graph
% corresponding to the decomposition of the input hypergraph according to
% J.A. Rodriguez. 
%% Syntax
% [adjMat, lapMat] = RodriguezLaplacian(HG);
%% Input
% HG - hypergraph object with incidence matrix property obj.IM 
%% Output
% * adjMat - adjacency matrix of the decomposed hypergraph
% * lapMat - graph Laplacian matrix of the decomposed hypergraph
%% Disclaimer
% The definition of Rodriguez's Laplacian from a hypergraph was taken from the
% below papers
% * Rodriguez, J. A. (2002). On the Laplacian eigenvalues and metric parameters of hypergraphs. Linear and Multilinear Algebra, 50(1), 1-14.
% * Rodriguez, J. A. (2003). On the Laplacian spectrum and walk-regular hypergraphs. Linear and Multilinear Algebra, 51, 285–297.
%% Code 
function [adjMat,lapMat] = RodriguezLaplacian(HG)
%RODRIGUEZLAPLACIAN Summary of this function goes here
%   Detailed explanation goes here
H = HG.IM;

adjMat=H*1*H';
adjMat=adjMat-diag(diag(adjMat));
Dvr=diag(sum(adjMat,2));
lapMat=Dvr-adjMat;
end

