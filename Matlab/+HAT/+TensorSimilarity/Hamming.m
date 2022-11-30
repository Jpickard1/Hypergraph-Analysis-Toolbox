function d = Hamming(A1, A2)
% A1 and A2 are adjacency tensors of two hypergraphs
% Returns the Hamming distance between two hypergraphs

sz = size(A1);
order = length(sz);
d = sum(abs(A1-A2), 'all')/(sz(1)^order-sz(1));

end
