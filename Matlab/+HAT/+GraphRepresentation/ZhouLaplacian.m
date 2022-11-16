%% Decompositions.GraphDecomp.ZhouLaplacian
% Returns the adjacency matrix and laplacian matrix of a graph
% corresponding to the decomposition of the input hypergraph according to
% Dengyong Zhou. 
%% Syntax
% [adjMat, lapMat] = ZhouLaplacian(HG);
%% Input
% HG - hypergraph object with incidence matrix property obj.IM 
%% Output
% * adjMat - adjacency matrix of the decomposed hypergraph
% * lapMat - graph Laplacian matrix of the decomposed hypergraph
%% Disclaimer
% The definition of Bolla's Laplacian from a hypergraph was taken from
% equation 3.3 of the below paper:
%
% Zhou, D., Huang, J., & Sch¨olkopf, B. (2005). Beyond pairwise classification and clustering using hypergraphs (Technical Report 143). Max Plank Institute for Biological Cybernetics, T¨ubingen, Germany.
%% Code 
function [adjMat,lapMat] = ZhouLaplacian(HG)
H = HG.IM;
eW = HG.edgeWeights;
de=sum(H,1)';

dv=H*eW;
dv(dv==0)=Inf; % convention
Dvinv=sparse(1:length(dv),1:length(dv),1./sqrt(dv),length(dv),length(dv));%diag(1./sqrt(dv))
Deinv=sparse(1:length(de),1:length(de),1./de,length(de),length(de));%diag(1./de);
W=sparse(1:length(eW),1:length(eW),eW,length(eW),length(eW));%diag(eW);
lapMat=eye(size(H,1))-Dvinv*H*W*Deinv*H'*Dvinv;
adjMat=H*W*Deinv*W*H';
end

