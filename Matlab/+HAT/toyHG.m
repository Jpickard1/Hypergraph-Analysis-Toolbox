function [HG] = toyHG(n, k, t)
%TOYHG This function returns a toy hypergraph
%
% PARAMETERS:
%   n: number of vertices
%   k: order of hypergraph
%   t: type of hypergraph: ['ring', 'chain', 'star','complete']
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: February 2023
    switch t
        case "hyperring"
            HG = HAT.simpleHypergraphs.hyperring(n, k);
        case "hyperchain"
            HG = HAT.simpleHypergraphs.hyperchain(n, k);
        case "hyperstar"
            HG = HAT.simpleHypergraphs.hyperstar(n, k);
        case "complete"
            HG = HAT.simpleHypergraphs.completeHG(n, k);
        otherwise
            error('invalid type of hypergraph')
    end
end

