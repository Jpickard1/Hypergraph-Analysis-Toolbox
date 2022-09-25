%% PHAT Application Note
%
%   This file contains code used to generate figures for the Bioinformatics
%   PoreC Hypergraph Analysis Toolbox Application Note
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: September 21, 2022

%% Preamble
clear all

%% 

I = randi([0 1], 10, 15)
H = Hypergraph('H', I)
ax = figure;
PlotIM.plotIncidenceMatrix(H, ax)
