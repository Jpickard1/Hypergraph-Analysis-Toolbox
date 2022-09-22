function r = sRadius(obj, s)
%SRADIUS Summary of this function goes here
%   Detailed explanation goes here
    I = obj.IM; 
    I(:, sum(I,1) < s) = 0; % remove edges that are smaller than s
    A = I*I'; % get clique-expanded adjacency matrix

    r = min(distances(graph(A), 'Method', 'unweighted'), 1);
end

