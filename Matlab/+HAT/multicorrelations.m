function [M, idxs] = multicorrelations(D, order, type)
%MULTIWAYCORRELATIONS This function computes multi-correlations among data
% * D is m x n data matrix
% * order is the number of variables considered for multicorrelations
% * type denotes the multicorrelation measures: 'Drezner', 'Wang', 'Taylor'
% Auth: Joshua Pickard

R = corrcoef(D);

[m, n] = size(D);
idxs = nchoosek(1:n, order);
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

