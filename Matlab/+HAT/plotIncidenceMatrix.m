function ax = plotIncidenceMatrix(HG, nodeshape, lineargs, show_h_bars, NameValueArgs)
%PLOT_INCIDENCE_MATRIX creates a new figure and plots the incidence matrix
%onto it. Returns the axes on which the incidence matrix is plotted. 
% HG: (n,m) double. Incidence matrix.
    
    arguments 
        HG(:,:)
        nodeshape = '.'
        lineargs = num2cell("-k")
        show_h_bars = true
        NameValueArgs.sort (1,1) logical = true
    end

    % The gray bars need to go underneath the line plot, which needs to go
    % underneath the scatter plot. But we need to make the scatter and line
    % plots before the bar plot use the padded ax.XLim to set the base of
    % the bar plot. So, we plot in this order:
    %   1. Scatter
    %   2. Line Plot
    %   3. Barh
    % Then we flip the graphics array of the axes.
    % 
    
    A = HG.IM;
    if NameValueArgs.sort
        A = HAT.SortIM.cardinalitySort(A);
        A = HAT.SortIM.min_max_sort(A);
    end

    ax = gca;
    ax.XLimitMethod = 'padded';
    ax.YLimitMethod = 'padded';

    %% Scatter
    s = HAT.ScatterIM.scatterIM(A, ax);
    s.Marker = nodeshape;
    s.CData = [0 0 0];
    hold on;

    %% Line Plot
    if class(lineargs) == "cell"
        p = HAT.LinePlotIM.lineplotIM(ax, A, lineargs);
    end

    %% Barh
    if show_h_bars
        HAT.BarhIM.barhIM(ax, A);
    end
    hold off;

    %% Axes
    ax.XLim = [0, size(A,2) + 1];
    ax.YLim = [0.5, size(A,1) + 0.5];
    ax.YDir = 'reverse';

    chi = get(ax, 'Children');
    set(ax, 'Children', flipud(chi));
end