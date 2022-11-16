%% HAT.matrixEntropy
% Computes the entropy score of a uniform hypergraph based on the
% generalized singular value decomposition of the Laplacian tensor of that hypergraph. 
%% Syntax
%   entropy = hypergraphEntropy(HG, false)
%% Input
% * HG - Hypergraph object. HG must represent a k-uniform hypergraph.
% * normalized - bool, indicated whether to normalize the Laplacian matrix or
% not. *Default*: false. 
%% Output
% * entropy - Entropy score for the hypergraph.
%% References
% * Dotson, Gabrielle A., et al. "Deciphering multi-way interactions in the human genome." Nature communications 13.1 (2022): 1-15.
% 
%% Code
function entropy = matrixEntropy(HG, normalized)
if nargin < 2
    normalized = false;
end
H = HG.IM;
H(sum(H, 2)==0, :) = []; 
D = diag(sum(H, 2));
E = diag(sum(H, 1));
if normalized
    L = eye(size(H, 1))-(D^(-1/2)*H*E^(-1)*H'*D^(-1/2));
else
    L = D-H*E^(-1)*H';
end
eigValues = eig(L);
nonzeroEigVal = eigValues(eigValues>1e-8);
normalizedEig = nonzeroEigVal/sum(nonzeroEigVal); 
entropy = -sum(normalizedEig.*log(normalizedEig));
end