%% Decompositions.TensorDecomp.adjacencyTensor
% 
%% Syntax
% 
%% Input
%
%% Output
% 
%% Disclaimer
%
%% Code 
function adjTensor = dualAdjacencyTensor(HG)
% The dual of a hypergraph is the hypergraph represented by the transposed
% incidence matrix.
Dual.IM = HG.IM';
Dual.edgeWeights = HG.nodeWeights;
adjTensor = adjacencyTensor(Dual);
end

