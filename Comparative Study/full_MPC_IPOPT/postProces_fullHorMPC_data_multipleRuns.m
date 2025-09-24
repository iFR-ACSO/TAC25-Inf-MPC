%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Safe-by-Design: Approximate Nonlinear Model 
%   Predictive Control with Realtime Feasibility" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: This script takes the fully stored workspace from
%   the simulation and computes the necessary ingredients for comparison
%   and also prepares the variables i.e. reduces the data.
%
%
%   Needed software: - CasADi 3.6 
%                    - A C/C++ compiler for .mex file generation (We use a
%                      MS 2022 C/C++ compiler) 
%
%
%   License: see License file in repository.   
%
% ------------------------------------------------------------------------
clc
clear

% load full workspace of full-horizon MPC script (might take some time)
load full_horizon_MPC_simulation_completeWS_3_runs.mat


% reduce data to a minimum just for direct comparison with
% infinitesimal-MPC
u_sol_vec_fullMPC = u_sol_all;
x_sol_vec_fullMPC = x_sol_all; 
tEnd_all_fullMPC  = tEnd_all; 
iter_all_fullMPC  = iter_all; 
Q_full = Q_weight;
R_full = R;


minSolveTime_full = min(tEnd_all_fullMPC, [], 'all') * 1000;
maxSolveTime_full = max(tEnd_all_fullMPC, [], 'all') * 1000;
% Convert matrix to cell array, one cell per row
tEnd_all_cell = mat2cell(tEnd_all_fullMPC, ones(1, size(tEnd_all_fullMPC, 1)), size(tEnd_all_fullMPC, 2));

% Remove NaNs and make each output a column vector
tEnd_all_cleaned = cellfun(@(row) row(~isnan(row))', tEnd_all_cell, 'UniformOutput', false);

% Concatenate all cleaned column vectors and compute the mean in ms
meanSolveTime_full = mean(cell2mat(tEnd_all_cleaned)) * 1000;



fprintf('Minimum solve time: %f ms\n', minSolveTime_full);
fprintf('Maximum solve time: %f ms\n', maxSolveTime_full);
fprintf('Mean solve time: %f ms\n', meanSolveTime_full);

% store for comparison in main folder
cd ..\
save('fullHor_MPC_comparison_3Runs.mat','meanSolveTime_full','maxSolveTime_full','minSolveTime_full','R_full','Q_full',...
     'tEnd_all_fullMPC','iter_all_fullMPC' ,'x_sol_vec_fullMPC','u_sol_vec_fullMPC','iter_all_fullMPC')
% come back
cd full_MPC_IPOPT\




