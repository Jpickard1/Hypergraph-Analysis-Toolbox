function s = colorByCardinality(s, A)
%COLORBYCARDINALITY Colors the edges of the incidence matrix by the number
%of vertices within each edge.
% s: scatter object representing the incidence matrix
% A: incidence matrix

[i, ~] = find(A);

colors = zeros(size(i));
cards = sum(A,1);
count = 1;
for index = 1:length(cards)
    for k = 1:(cards(index))
        colors(count) = cards(index);
        count = count + 1;
    end
end

% s.CData = colors(idx); 
s.CData = colors;
end

