%% Decompositions.GraphDecomp.cliqueGraph
% Returns the adjacency matrix and graph Laplacian of the clique-expansion
% of the input hypergraph.
%% Syntax
%  [adjMat, lapMat] = cliqueGraph(HG)
%% Input
% HG - hypergraph object with incidence matrix property obj.IM
%% Output
% * adjMat - adjacency matrix of the clique expansion
% * lapMat - graph Laplacian matrix of the clique expansion
%% Disclaimer
%
%% Code 
function [adjMat,lapMat] = cliqueGraph(HG)
H = HG.IM;
eW = HG.edgeWeights;
de=sum(H,1)';

adjMat=H*sparse(1:length(eW),1:length(eW),eW,length(eW),length(eW))*H';
adjMat=adjMat-diag(diag(adjMat));
eW1=(de-1).*eW;
dvc=H*eW1; %this should be same as sum(adjMat,2)
dvc(dvc==0)=Inf; % convention
Dvc=sparse(1:length(dvc),1:length(dvc),1./sqrt(dvc),length(dvc),length(dvc));%diag(1./sqrt(dvc)))
lapMat=eye(size(Dvc))-Dvc*adjMat*Dvc; % normalized Laplacian
end

