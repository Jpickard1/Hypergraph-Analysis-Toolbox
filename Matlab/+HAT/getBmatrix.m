function B = getBmatrix(v, n)

B = zeros(n, length(v));
for i = 1:length(v)
    B(v(i), i) = 1;
end