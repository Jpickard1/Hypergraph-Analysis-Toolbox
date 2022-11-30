function p = lineplotIM(ax, A, lineargs)
%LINEPLOTIM Summary of this function goes here
%   Detailed explanation goes here
    [m, n] = size(A);
    Abin = ones(m, n);
    Abin(A == 0) = 0;
    
    % incidence matrix with nz entries scaled to node number.
    Y = (Abin' * sparse(1:m, 1:m, 1:m, m, m))'; 

    % matrix holding x-values associated with each hyperedge plot
    X = ndgrid(1:n, 1:m)';
    
    % some cell array manipulations!
    Ycell = mat2cell(Y', ones(size(Y,2),1));
    Xcell = mat2cell(X', ones(size(X,2),1));
    
    % remove zero entries from each cell
    for idx = 1:size(Ycell,1)
        logical_idx = Ycell{idx} == 0;
        Ycell{idx}(logical_idx) = [];
        Xcell{idx}(logical_idx) = [];
    end
    
    % a cell array with plotting triplets:
    % 1. XData
    % 2. YData
    % 3. lineargs
    plottingCell = cell(3*size(Y,2),1);
    plottingCell(1:3:end) = Xcell;
    plottingCell(2:3:end) = Ycell;
    plottingCell(3:3:end) = lineargs;
    
    p = plot(ax, plottingCell{:});
end

