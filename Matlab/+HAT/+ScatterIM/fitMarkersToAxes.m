function ax = fitMarkersToAxes(s, ax)
%FITMARKERSTOAXES Adjusts the SizeData property of the scatter object s to
%fit withing the axes ax. gca is passed in by default.
    arguments 
        s 
        ax = gca
    end

    marker = s.Marker;

    % these are all the MATLAB built-in markers for scatter.
    small = '.';
    medium = ['p', 'h', 'x', '|'];
    large = ['s', 'd', '^', '<', '>', 'v', '_', '*', '+'];

    currentunits = get(ax,'Units');
    set(ax, 'Units', 'Points');
    axpos = get(ax,'Position'); % [left bottom width height]
    set(ax, 'Units', currentunits);
    
    space = min(axpos(3)/diff(ax.XLim), axpos(4)/diff(ax.YLim));
    
    if ismember(marker, small)
        scale_marker = 2;
    elseif ismember(marker, medium)
        scale_marker = 0.8;
    else
        scale_marker = 0.72;
    end

    area = pi*(scale_marker*space/(2))^2;
    set(s, 'SizeData', area);
end

