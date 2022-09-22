%% Computations.clusteringCoefficient
% Computes the generalized clustering coefficient on a k-uniform hypergraph.
%% Syntax
%   p = clusteringCoefficient(HG)
%% Input 
% * HG - Hypergraph object. HG must represent a k-uniform hypergraph.
%% Output
% * p - Average clustering coefficient of HG.
%% Disclaimer
% The formula for average distance was obtained from equation 31 of the
% paper below.
%
% Amit Surana, Can Chen, and Indika Rajapakse. "Hypergraph dissimilarity measures." arXiv preprint arXiv:2106.08206 (2021).
%% Code
function p = clusteringCoefficient(HG)
import Decompositions.tensor.uniformEdgeSet
A = uniformEdgeSet(HG);
n = size(A,2);

pp = zeros(n, 1);
for i = 1:n
    [id1, ~] = find(A == i);
    N = unique(A(id1, :));
    N(N == i) = [];
    C = nchoosek(N, size(A, 2));
    C = sort(C, 2);
    A = sort(A, 2);
    [~, loc] = ismember(C, A, 'rows');
    q = length(find(loc>0));
    pp(i) = q/size(C, 1);
end 

pp(isnan(pp))=0;
p = mean(pp);

end

