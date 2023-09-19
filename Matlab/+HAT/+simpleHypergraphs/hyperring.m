function [HG] = hyperring(V, k)
%HYPERRING This function returns a hyperring hypergraph
%
%   See definition 3 in Controllability of Hypergraphs
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: February 20, 2023

if V == k
    IM = ones(V,1);
    HG = Hypergraph('IM', sparse(IM));
    return
end

IM = zeros(V, V);
for e=1:V-k+1
    IM(e:e+k-1, e) = 1;
end
for e=V-k+2:V
    f = 1:e-(V-k+2-1);
    b = e:V;
    IM([f b] ,e) = 1;
end
HG = Hypergraph('IM', sparse(IM));

end

