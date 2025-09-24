
clc
clear

% load full workspace of full-horizon MPC script (might take some time)
load full_horizon_MPC_simulation_completeWS.mat


% reduce data to a minimum just for direct comparison with
% infinitesimal-MPC
u_sol_vec_fullMPC = u_sol_all;
x_sol_vec_fullMPC = x_sol_all; 
Q_full = Q;
R_full = R;


iter_conv_full = k;

minSolveTime_full = min(tEnd, [], 'all') * 1000;
maxSolveTime_full = max(tEnd, [], 'all') * 1000;
meanSolveTime_full = mean(tEnd, 'all') * 1000;

fprintf('Minimum solve time: %f ms\n', minSolveTime_full);
fprintf('Maximum solve time: %f ms\n', maxSolveTime_full);
fprintf('Mean solve time: %f ms\n', meanSolveTime_full);

% store for comparison in main folder
cd ..\
save('fullHor_MPC_comparison.mat',"meanSolveTime_full","maxSolveTime_full","minSolveTime_full","R_full","Q_full","x_sol_vec_fullMPC","u_sol_vec_fullMPC","iter_conv_full")
% come back
cd full_MPC\




