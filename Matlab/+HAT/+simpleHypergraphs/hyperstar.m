function [HG] = hyperstar(V, k)
%HYPERRING This function returns a hyperstar hypergraph
%
%   See definition 4 in Controllability of Hypergraphs
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: February 20, 2023

IM = zeros(V, V-k+1);
for e=1:V-k+1
    IM([1:k-1 (k-1+e)], e) = 1;
end
HG = Hypergraph('IM', IM);

end

