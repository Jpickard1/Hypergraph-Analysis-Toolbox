function D = degreeTensor(HG)
%DEGREETENSOR Summary of this function goes here
%   Detailed explanation goes here
H = HG.IM;
eW = HG.edgeWeights;
numNodes = size(H, 1);
order = sum(H(:,1), 1);

% Used to generate linear indices
p = cumprod([1 (numNodes * ones(1, order-1))]);

d = sum(H,1) .* eW;
D = zeros(numNodes*ones(1,order));
for i=1:numNodes
    LINIDX = ((i*ones(1,order))-1)*p(:)+1;
    D(LINIDX) = d(i);
end

end

