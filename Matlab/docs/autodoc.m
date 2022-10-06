%% Hypergraph Analysis Toolbox

% <include>../+Computations/averageDistance.m</include>
% <include>../+Computations/clusteringCoefficient.m</include>
% <include>../+Computations/hypergraphCentrality.m</include>
% <include>../+Computations/hypergraphEntropy.m</include>

ls('..')

files = dir(fullfile('..', '*.m'));

filelist = dir(fullfile('..', '**\*.*'));  %get list of files and folders in any subfolder
filelist = filelist(~[filelist.isdir])

for f=1:length(filelist)
    fil = filelist(f).name
    if strcmp(fil(length(fil) - 1:length(fil)), '.m')
        publish(filelist(f).folder + "\" + filelist(f).name, 'format', 'pdf', 'outputDir', 'C:\Joshua\Hypergraph-Analysis-Toolbox\Matlab\docs\singleFiles\', 'evalCode', false);
    end
end
