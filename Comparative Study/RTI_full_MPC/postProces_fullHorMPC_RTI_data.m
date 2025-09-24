%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
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
load full_horizon_MPC_simulation_completeWS_3_runs_RTI.mat


% reduce data to a minimum just for direct comparison with
% infinitesimal-MPC
u_sol_vec_fullMPC_RTI = u_sol_all;
x_sol_vec_fullMPC_RTI = x_sol_all; 
tEnd_all_fullMPC_RTI  = tEnd_all; 
iter_all_fullMPC_RTI  = iter_all; 
Q_full_RTI = Q_weight;
R_full_RTI = R;


minSolveTime_full_RTI = min(tEnd_all_fullMPC_RTI, [], 'all') * 1000;
maxSolveTime_full_RTI = max(tEnd_all_fullMPC_RTI, [], 'all') * 1000;
% Convert matrix to cell array, one cell per row
tEnd_all_cell = mat2cell(tEnd_all_fullMPC_RTI, ones(1, size(tEnd_all_fullMPC_RTI, 1)), size(tEnd_all_fullMPC_RTI, 2));

% Remove NaNs and make each output a column vector
tEnd_all_cleaned = cellfun(@(row) row(~isnan(row))', tEnd_all_cell, 'UniformOutput', false);

% Concatenate all cleaned column vectors and compute the mean in ms
meanSolveTime_full_RTI = mean(cell2mat(tEnd_all_cleaned)) * 1000;

fprintf('Minimum solve time: %f ms\n', minSolveTime_full_RTI);
fprintf('Maximum solve time: %f ms\n', maxSolveTime_full_RTI);
fprintf('Mean solve time: %f ms\n', meanSolveTime_full_RTI);

% store for comparison in main folder
cd ..\
save('fullHor_MPC_RTI_comparison_3Runs.mat','meanSolveTime_full_RTI','maxSolveTime_full_RTI','minSolveTime_full_RTI','R_full_RTI','Q_full_RTI',...
     'tEnd_all_fullMPC_RTI','iter_all_fullMPC_RTI' ,'x_sol_vec_fullMPC_RTI','u_sol_vec_fullMPC_RTI','iter_all_fullMPC_RTI')

% come back
cd RTI_full_MPC\




