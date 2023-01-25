function [N, DS] = load(dataset)
%LOAD Returns a builtin dataset of HAT
%
% Parameters:
%   * dataset: which dataset to load
%
% Returns:
%   * N: network structure as either incidence matrix for hypergraph or
%   adjaccency matrix for graphs
%   * DS: additional information about the network that may be relevant to
%   working with this data
%
% Auth: Joshua Pickard
% Date: November 28, 2022

[current_path,~,~] = fileparts(mfilename('fullpath'))
len = length(current_path);
current_path(len-3:end) = [];
dp = string(current_path) + "+Data/";

switch dataset
    case 'karate'
        load(dp + "karate.mat");
        DS = Problem;
        N = DS.A;
    case 'ArnetMiner Citation'
        load(dp + "aminer_cocitation.mat");
        N = S;
        DS = [];
    case 'ArnetMiner Reference'
        load(dp + "aminer_coreference.mat");
        N = S;
        DS = [];
    case 'Citeseer Citation'
        load(dp + "citeseer_cocitation.mat");
        N = S;
        DS = [];
    case 'Citeseer Reference'
        load(dp + "citeseer_coreference.mat");
        N = S;
        DS = [];
    case 'Cora Citation'
        load(dp + "cora_cocitation.mat");
        N = S;
        DS = [];
    case 'Cora Reference'
        load(dp + "cora_coreference.mat");
        N = S;
        DS = [];
    case 'DBLP'
        load(dp + "dblp.mat");
        N = S;
        DS = [];
end

end


