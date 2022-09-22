function [bins, binSize] = sConnectedComponents(obj, s, outputForm)
%SCONNECTEDCOMPONENTS Returns the s-connected components of the hypergraph
%specified by the incidence matrix.
%   
    arguments
        obj
        s = 1
        outputForm = "vector"
    end
    I = obj.IM; 
    I(:, sum(I,1) < s) = 0; % remove edges that are smaller than s
    A = I*I'; % get clique-expanded adjacency matrix

    [bins, binSize] = conncomp(graph(A), "OutputForm", outputForm);
end

