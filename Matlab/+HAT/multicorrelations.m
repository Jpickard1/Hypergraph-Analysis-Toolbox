function [M, idxs] = multicorrelations(D, order, type, idxs)
%MULTIWAYCORRELATIONS This function computes multi-correlations among data
% * D is m x n data matrix
% * order is the number of variables considered for multicorrelations
% * type denotes the multicorrelation measures: 'Drezner', 'Wang', 'Taylor'
% * idxs denotes pairs of indices to have mutli-correlations computed to
%   avoid the combinatorial complexity of trying all n choose k pairs.
% Auth: Joshua Pickard
% Date: November 2022

R = corrcoef(D);

[m, n] = size(D);
if nargin == 3
    idxs = nchoosek(1:n, order);
end

M = zeros(length(idxs), 1);

if strcmp(type, 'Taylor')
    w = (1/sqrt(order));
end

for i=1:length(idxs)
    minor = R(idxs(i,:), idxs(i,:));
    if strcmp(type,'Drezner')
        M(i) = 1 - min(eig(minor));
    elseif strcmp(type, 'Wang')
        M(i) = (1-det(minor))^0.5;
    elseif strcmp(type, 'Taylor')
        M(i) = w * std(eig(corrcoef(minor)));
    end
end

end

