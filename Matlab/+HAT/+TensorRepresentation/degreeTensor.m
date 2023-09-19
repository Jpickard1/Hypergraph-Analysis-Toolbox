function D = degreeTensor(HG)
%DEGREETENSOR Generates degree tensor
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: April 17, 2023
H = HG.IM;
eW = HG.edgeWeights;
numNodes = size(H, 1);
order = sum(H(:,1), 1);

% Used to generate linear indices
p = cumprod([1 (numNodes * ones(1, order-1))]);

d = H * eW;
% d = sum(H,2) .* eW;
D = zeros(numNodes*ones(1,order));
for i=1:numNodes
    LINIDX = ((i*ones(1,order))-1)*p(:)+1;
    D(LINIDX) = d(i);
end

end

