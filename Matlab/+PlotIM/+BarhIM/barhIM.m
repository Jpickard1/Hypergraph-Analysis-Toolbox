function b = barhIM(ax, A)
%barhIM Plots horizontal stripes for aesthetic value. 
%   

    [m, ~] = size(A);

    currXMethod = ax.XLimitMethod;
    currYMethod = ax.YLimitMethod;

    ax.XLimitMethod = 'tight';
    ax.YLimitMethod = 'tight';  
    
    bars = 1:2:m;
    b = barh(ax, bars, (ax.XLim(2) + 1)*ones(size(bars)));
    b.FaceColor = [0.3 0.3 0.3];
    b.FaceAlpha = 0.15;
    b.BarWidth = 0.5;
    b.EdgeAlpha = 0;
    b.BaseValue = ax.XLim(1);
    
    if mod(m, 2) == 0
        b2 = barh(ax, m, ax.XLim(2));
        b2.BarWidth = 1;
        b2.FaceAlpha = 0;
        b2.EdgeAlpha = 0;
        b2.BaseValue = ax.XLim(1);
    end

    ax.XLimitMethod = currXMethod;
    ax.YLimitMethod = currYMethod;
end

