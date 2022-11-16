function [outputArg1,outputArg2] = vxClusterCoef(HG, vx)
%VXCLUSTERCOEF This function computes the clustering coefficient of a
%   single hypergraph vertex. 
%
% Auth: Joshua Pickard
%       jpic@umich.edu

IM = HG.IM;
neightbors = find(IM(:,vx) == 1)

end

