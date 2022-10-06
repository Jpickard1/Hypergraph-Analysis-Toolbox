function d = SpectralS(A1,A2)
% A1 and A2 are adjacency tensors of two hypergraphs
% Returns the spectral distance (based on H-eigenvalue)
% between two hypergraphs

sz = size(A1);
numNodes = sz(1);
order = length(sz);

D1 = zeros(numNodes^order, 1);
D2 = zeros(numNodes^order, 1);
for i = 1:numNodes
    D1(indexMapping(sz, i*ones(1, order))) = sum(A1(i, :), 'all');
    D2(indexMapping(sz, i*ones(1, order))) = sum(A2(i, :), 'all');    
end

D1 = reshape(D1, ones(1, order)*numNodes);
D2 = reshape(D2, ones(1, order)*numNodes);

L1 = reshape(D1-A1, sz);
L2 = reshape(D2-A2, sz);

% JP - Joshua changed the below 2 lines from heig to height. It was not
% working previously. JP - heig is from a matlab toolbox. This change was
% undone
heig1 = uniquetol(heig(L1)', 1e-4);
heig2 = uniquetol(heig(L2)', 1e-4);

if length(heig1) <= length(heig2)
    heig1 = [zeros(length(heig2)-length(heig1), 1); heig1];
else
    heig2 = [zeros(length(heig1)-length(heig2), 1); heig2];
end

heig1 = heig1./sum(heig1);
heig2 = heig2./sum(heig2);
    
d = sum(abs(heig1-heig2).^2);
end

function linearIdx = indexMapping(sz, subscript)
fisrtSub = subscript(1);
subscript = subscript(2:end)-1;
szProd = zeros(1, length(sz)-1);
for i = 1:length(sz)-1
    szProd(i) = prod(sz(1:i));
end
linearIdx = fisrtSub+sum(subscript.*szProd);
end

