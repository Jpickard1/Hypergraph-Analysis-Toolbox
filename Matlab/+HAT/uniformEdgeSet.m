%% Decompositions.TensorDecomp.uniformEdgeSet
% takes the incidence matrix of a hypergraph and returns a matrix
% representing the hyperedge set of the hypergraph. Each row is a
% hyperedge. 
%% Syntax
% 
%% Input
%
%% Output
% 
%% Disclaimer
%
%% Code 
function hyperedgeSet = uniformEdgeSet(HG)
H = HG.IM;
k = sum(H(:,1), 1);
hyperedgeSet=ones(size(H,2), k);
for i=1:size(H,2)
   e=H(:,i);
   ind=find(e>0);
   hyperedgeSet(i, :) = ind; 
end

end

