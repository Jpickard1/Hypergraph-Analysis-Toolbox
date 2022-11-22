% Constructor
I = randi([0,1], 5, 10)
HG = Hypergraph('IM',I)

% Visualize
p = HG.plot()

% Tensor representations
I = [1 1 1 0;
     1 1 0 1;
     1 0 1 1;
     0 1 1 1];

HG = Hypergraph('IM',I)
A = HG.adjTensor
D = HG.degreeTensor
L = HG.laplacianTensor

% Graph Expansions
I = [1 1 1 0;
     1 1 0 1;
     1 0 1 1]
HG = Hypergraph('IM',I)

C = full(HG.cliqueGraph())
S = full(HG.starGraph())
L = full(HG.lineGraph())

% Laplacians
[A1, L1] = (HG.laplacianMatrix("Bolla"))
[A2, L2] = (HG.laplacianMatrix("Rodriguez"))
[A3, L3] = (HG.laplacianMatrix("Zhou"))

% Hypergraph Similarity
HG1 = HAT.uniformErdosRenyi(6,8,3);
HG2 = HAT.uniformErdosRenyi(6,8,3);

D1 = HAT.directSimilarity(HG1,HG2,'Hamming')
D2 = HAT.directSimilarity(HG1,HG2,'Spectral-S')
D3 = HAT.directSimilarity(HG1,HG2,'Spectral-H')
D4 = HAT.directSimilarity(HG1,HG2,'Centrality')

A1 = HG1.cliqueGraph();
A2 = HG2.cliqueGraph();

ID1 = HAT.indirectSimilarity(A1, A2, 'type', 'centrality')
ID2 = HAT.indirectSimilarity(A1, A2, 'type', 'Hamming')
ID3 = HAT.indirectSimilarity(A1, A2, 'type', 'Jaccard')
ID4 = HAT.indirectSimilarity(A1, A2, 'type', 'deltaCon')
ID5 = HAT.indirectSimilarity(A1, A2, 'type', 'heatKer')
ID6 = HAT.indirectSimilarity(A1, A2, 'type', 'spanTree')
ID7 = HAT.indirectSimilarity(A1, A2, 'type', 'Spectral')

% Entropy
HG = HAT.uniformErdosRenyi(6,8,3);
E = HG.tensorEntropy()
M = HG.matrixEntropy()

% Controllability
I = randi([0,1], 5, 10);
HG = Hypergraph('IM',I);
B = HAT.ctrbk(HG,[1])

% Other Properties
HG = HAT.uniformErdosRenyi(6,8,3);
avg = HG.avgDistance
clusterCoef = HG.clusteringCoef
centrality = HAT.centrality(HG)

%% Python Comparison
IM = [1 1 0;
      1 1 0;
      1 0 1;
      0 1 1;
      0 0 1];
HG = Hypergraph('IM',IM)
[A1, L1] = (HG.laplacianMatrix("Bolla"))
[A2, L2] = (HG.laplacianMatrix("Rodriguez"))
[A3, L3] = (HG.laplacianMatrix("Zhou"))

A = HG.adjTensor
HAT.directSimilarity(HG, HG, 'Spectral-S')

%% Doc 2
% 1. Construction
IM = [1 1 0;
      1 1 0;
      1 0 1;
      0 1 1;
      0 0 1];
HG = Hypergraph('IM',IM)

% 2. Visualization
HG.plot()

% 3. Expansion
C = HG.cliqueGraph;
figure; plot(graph(C));
title('Clique Expansion');

S = HG.starGraph;
figure; plot(graph(S));
title('Star Expansion');

% 4. Structural Properties
avgDist = HG.avgDistance
clusterCoef = HG.clusteringCoef
C = HG.centrality()

% 5. Similarity
IM = [1 1 0;
      0 1 1;
      1 0 0;
      0 1 1;
      1 0 1];
HG2 = Hypergraph('IM',IM)
D1 = HAT.directSimilarity(HG, HG2, 'Hamming')
D2 = HAT.directSimilarity(HG, HG2, 'Centrality')
D3 = HAT.directSimilarity(HG, HG2, 'Spectral-S')
D4 = HAT.directSimilarity(HG, HG2, 'Spectral-H')

I1 = HAT.indirectSimilarity(HG.cliqueGraph, HG2.cliqueGraph, "type",1)

% 6. Controllability
B = HG.ctrbk([1  3])

% 7. Multicorrelation
%       6 rvars. 8 measurements
D = rand(8, 6);
[M, sets] = HAT.multicorrelations(D, 3, 'Drezner');

