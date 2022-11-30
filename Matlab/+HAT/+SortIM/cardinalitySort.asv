function An = cardinalitySort(A, direction)
%CARDINALITYSORT Returns a new incidence matrix whose edges are sorted by
%cardinality. 
arguments
    A (:,:)
    direction = 'descend'
end

cardinality = sum(A,1);
[~, idx] = sort(cardinality, direction);
An = A(:, idx);

end

