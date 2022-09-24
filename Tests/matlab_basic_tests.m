%% MATLAB Basic Tests
%   This script is intended to test the ability to call every function in the
%   package as intended.
%   
%   NOTE: This script does not test that functions produce the intended
%   output.
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: September 23, 2022

%% Preamble
clear all;
close all;
% I = randi([0 1], 5, 10)

I = [   1     1     1     1     1     1     0     0     0     0;
        1     1     1     0     0     0     1     1     1     0;
        1     0     0     1     1     0     1     1     0     1;
        0     1     0     1     0     1     1     0     1     1;
        0     0     1     0     1     1     0     1     1     1;]

%% Hypergraph dir
H = Hypergraph('H', I);             % Constructor

[a,b] = sConnectedComponents(H, 1)  % Function
a = sRadius(H, 1)                   % Function

%% Computations dir
[d, dmax] = Computations.averageDistance(H)
p = Computations.clusteringCoefficient(H)

% This function is curently broken
% [nodeCentrality, edgeCentrality] = Computations.hypergraphCentrality(H)
entropy = Computations.hypergraphEntropy(H);

%% Decompositions

% Tensor decompositions
A = Decompositions.TensorDecomp.adjacencyTensor(H);
A = Decompositions.TensorDecomp.dualAdjacencyTensor(H);
hyperedgeSet = Decompositions.TensorDecomp.uniformEdgeSet(H);
E = Decompositions.TensorDecomp.edgeSetToIncidenceMatrix(hyperedgeSet);
% This function should be refactored to accept the hypergraph object rather
% than the hyperedgeSet

% Graph decompositions
[adjMat,lapMat] = Decompositions.GraphDecomp.BollaLaplacian(H);
[adjMat,lapMat] = Decompositions.GraphDecomp.cliqueGraph(H);
[adjMat,lapMat] = Decompositions.GraphDecomp.dualGraph(H);
[adjMat,lapMat] = Decompositions.GraphDecomp.lineGraph(H);
[adjMat,lapMat] = Decompositions.GraphDecomp.RodriguezLaplacian(H);
[adjMat,lapMat] = Decompositions.GraphDecomp.starGraph(H);
[adjMat,lapMat] = Decompositions.GraphDecomp.ZhouLaplacian(H);

%% Dissimilarity measures
d = DissimilarityMeasures.graphDissimilarity(randi([0 1], 5, 5), randi([0 1], 5, 5));
d = DissimilarityMeasures.hypergraphDissimilarity(randi([0 1], 5, 5), randi([0 1], 5, 5));

% Tensor Dissimilarity
A1 = randi([0 1], 5, 5, 5);
A2 = randi([0 1], 5, 5, 5);
d = DissimilarityMeasures.TensorDis.Hamming(A1, A2);
d = DissimilarityMeasures.TensorDis.SpectralH(A1, A2);
d = DissimilarityMeasures.TensorDis.SpectralS(A1, A2);

%% Controllability
