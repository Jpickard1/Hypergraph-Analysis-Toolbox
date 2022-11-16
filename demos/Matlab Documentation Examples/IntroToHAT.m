% 1. Construction
IM = [1 1 1 0 0;
      1 1 0 1 0;
      0 0 1 1 1];
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
C = HG.centrality

% 5. Similarity
% 6. Controllability
% 7. Multicorrelation