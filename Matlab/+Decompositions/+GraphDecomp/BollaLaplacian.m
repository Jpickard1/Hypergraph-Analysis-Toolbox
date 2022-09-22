%% Decompositions.GraphDecomp.BollaLaplacian
% Returns the adjacency matrix and laplacian matrix of a graph
% corresponding to the decomposition of the input hypergraph according to
% Mariana Bolla. 
%% Syntax
% [adjMat, lapMat] = BollaLaplacian(HG);
%% Input
% HG - hypergraph object with incidence matrix property obj.IM 
%% Output
% * adjMat - adjacency matrix of the decomposed hypergraph
% * lapMat - graph Laplacian matrix of the decomposed hypergraph
%% Disclaimer
% The definition of Bolla's Laplacian from a hypergraph was taken from the
% below paper.
%
% Bolla, M. (1993). Spectra, euclidean representations and clusterings of hypergraphs. Discrete Mathematics, 117.
function [adjMat,lapMat] = BollaLaplacian(HG)
%BOLALAPLACIAN Summary of this function goes here
%   Detailed explanation goes here
H = HG.IM;
de=sum(H,1)';
H=H(:,de>1); % remove edges which represent self loops or empty
de=sum(H,1)';

Deinv=sparse(1:length(de),1:length(de),1./de,length(de),length(de)); %diag(1./de)
adjMat=H*Deinv*H';
dv=sum(H,2); %this should be same as sum(adjMat,2)
Dv=sparse(1:length(dv),1:length(dv),dv,length(dv),length(dv));
lapMat=Dv-adjMat;
end

