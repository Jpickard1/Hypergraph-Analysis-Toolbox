function s = colorNodesByDegree(s, A)
%COLORNODESBYDEGREE Colors each node of the incidence matrix by the number
% edges it is present in.
% s: scatter object representing the incidence matrix
% A: incidence matrix

[i, j] = find(A);
[~, idx] = sort(i);
j = j(idx);
[~, idx] = sort(j);

colors = zeros(size(j));
cards = sum(A,2);
count = 1;
for index = 1:length(cards)
    for k = 1:(cards(index))
        colors(count) = cards(index);
        count = count + 1;
    end
end

s.CData = colors(idx); 
end

