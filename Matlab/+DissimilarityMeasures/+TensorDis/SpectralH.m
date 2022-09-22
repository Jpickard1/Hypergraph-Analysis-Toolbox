function d = SpectralH(A1,A2)
% A1 and A2 are adjacency tensors of two hypergraphs
% Returns the spectral distance (based on generalized singular values)
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

Lm1 = reshape(D1-A1, sz(1), sz(1)^(order-1));
Lm2 = reshape(D2-A2, sz(1), sz(1)^(order-1));

s1 = svd(Lm1, 'econ');
s2 = svd(Lm2, 'econ');

s1 = s1./sum(s1);
s2 = s2./sum(s2);

d = sum(abs(s1-s2).^2)/length(s1);
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
