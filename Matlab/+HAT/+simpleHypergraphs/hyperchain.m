function [HG] = hyperchain(V, k)
%HYPERRING This function returns a hyperchain hypergraph
%
%   See definition 2 in Controllability of Hypergraphs
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: February 20, 2023

IM = zeros(V, V-k+1);
for e=1:V-k+1
    IM(e:e+k-1, e)=1;
end
HG = Hypergraph('IM', sparse(IM));

end

