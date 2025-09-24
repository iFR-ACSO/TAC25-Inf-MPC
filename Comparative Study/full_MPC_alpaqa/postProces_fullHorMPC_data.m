
clc
clear

% load full workspace of full-horizon MPC script (might take some time)
load full_horizon_MPC_simulation_completeWS_alpaqa.mat


% reduce data to a minimum just for direct comparison with
% infinitesimal-MPC
u_sol_vec_fullMPC_alpaqa = u_sol_vec;
x_sol_vec_fullMPC_alpaqa = x_sol_vec; 
Q_full_alpaqa = Q;
R_full_alpaqa = R;


iter_conv_full_alpaqa = k;

minSolveTime_full_alpaqa = min(tEnd, [], 'all') * 1000;
maxSolveTime_full_alpaqa = max(tEnd, [], 'all') * 1000;
meanSolveTime_full_alpaqa = mean(tEnd, 'all') * 1000;

fprintf('Minimum solve time: %f ms\n', minSolveTime_full_alpaqa);
fprintf('Maximum solve time: %f ms\n', maxSolveTime_full_alpaqa);
fprintf('Mean solve time: %f ms\n', meanSolveTime_full_alpaqa);

% store for comparison in main folder
cd ..\
save('fullHor_MPC_comparison_alpaqa.mat',"meanSolveTime_full_alpaqa","maxSolveTime_full_alpaqa","minSolveTime_full_alpaqa","R_full_alpaqa","Q_full_alpaqa","x_sol_vec_fullMPC_alpaqa","u_sol_vec_fullMPC_alpaqa","iter_conv_full_alpaqa")
% come back
cd full_MPC_alpaqa\




