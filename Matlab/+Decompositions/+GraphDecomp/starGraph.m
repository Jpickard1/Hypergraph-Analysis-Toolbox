%% Decompositions.GraphDecomp.starGraph
% Returns the adjacency matrix and graph Laplacian of the star-expansion
% of the input hypergraph.
%% Syntax
%  [adjMat, lapMat] = starGraph(HG)
%% Input
% HG - hypergraph object with incidence matrix property obj.IM
%% Output
% * adjMat - adjacency matrix of the star expansion
% * lapMat - graph Laplacian matrix of the starexpansion
%% Disclaimer
%
%% Code
function [adjMat ,lapMat] = starGraph(HG)
adjMat = [];

H = HG.IM;
de=sum(H,1)';
H=H(:,de>1); % remove edges which represent self loops or empty
eW = HG.edgeWeights;
de=sum(H,1)';

des=eW;
eW1=eW./de;
dvs=H*eW1;
dvs(dvs==0)=Inf; % convention
Dvs=sparse(1:length(dvs),1:length(dvs),1./sqrt(dvs),length(dvs),length(dvs)); 
Des=sparse(1:length(des),1:length(des),1./sqrt(des),length(des),length(des));
A=Dvs*H*sparse(1:length(eW),1:length(eW),eW,length(eW),length(eW))*Des;
lapMat=A*A'; %normalized projected Laplacian
end

