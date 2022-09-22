function An = select_by_node_presence(A, nodes, union)
%Returns a subset of the incidence matrix with hyperedges that include 
%nodes of interest. If one node is of interest, this function
%selects all hyperedges that include that node. If more than one node is of
%interest, the user can specify whether this function returns all
%hyperedges that contain ALL of the nodes of interest or all hyperedges 
% that contain ANY of the nodes of interest. 
%
% 
% A: (m,n) double. Incidence matrix.
% nodes: (1,k) double. Indices of nodes of interest, where k <= m and each
%        entry in "nodes" <= m.
%        Default: 1:m (all nodes)
% uniion: logical. Specifies selection of hyperedges. If true, keep 
%        hyperedges that contain all nodes in "nodes". If false, keep 
%        hyperedges that contain any node in "nodes".
%        Default: true
                
    arguments
        A (:,:) 
        nodes (1,:) = 1:size(A,1)
        union logical = true
    end

    % (n,k) logical array with ones indicating node presence in a hyperedge
    V = A(nodes, :) > 0;

    if union
        % (n,1) logical with ones for columns of V that are all ones
        idx = all(V, 1);
    else
        % (n,1) logical with ones for columns of V that contain any ones
        idx = any(V, 1);
    end

    An = A(:, idx);
end
