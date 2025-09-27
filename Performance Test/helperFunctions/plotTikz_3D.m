%% ------------------------------------------------------------------------
%
%   Supplementary Material for "Safe-by-Design: Approximate Nonlinear Model 
%   Predictive Control with Realtime Feasibility" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: This script is used to evaluate the second test case
%   in [1]. Plots are generated, the performance is evaluated and the plots
%   (and if desired) the tikz pictures found in [1] are generated.
%
%   
%   References: [1] Olucak, J, de Oliveira, A.C.B. and Cunis, T - 
%   Infinitesimal-horizon model predictive control as control barrier and 
%   Lyapunov function approach, submitted to IEEE Transaction on Automatic
%   Control
%
%   License: see License file in repository.   
%
% ------------------------------------------------------------------------

close all
% clear
clc


addpath("helperFunctions\")

% load MC data
% load 'full_WS_KeepOutResults_withCompFun.mat'



%% Compute statistics on computation time in miliseconds
minSolveTime_infMPC  = min(tEnd_all, [], 'all') * 1000;
maxSolveTime_infMPC  = max(tEnd_all, [], 'all') * 1000;
% Convert matrix to cell array, one cell per row
tEnd_all_cell = mat2cell(tEnd_all, ones(1, size(tEnd_all, 1)), size(tEnd_all, 2));

% Remove NaNs and make each output a column vector
tEnd_all_cleaned = cellfun(@(row) row(~isnan(row))', tEnd_all_cell, 'UniformOutput', false);

% Concatenate all cleaned column vectors and compute the mean in ms
meanSolveTime_infMPC = mean(cell2mat(tEnd_all_cleaned)) * 1000;

fprintf('Minimum solve time: %f ms\n', minSolveTime_infMPC);
fprintf('Maximum solve time: %f ms\n', maxSolveTime_infMPC);
fprintf('Mean solve time: %f ms\n', meanSolveTime_infMPC);

% Compute statistics on sufficient condition
minSuffCond = min(suffCon_all, [], 'all'); % just for plotting
maxSuffCond = max(suffCon_all, [], 'all');

% this value should be <= 0 
fprintf('Maximum value of sufficient condition: %f\n', maxSuffCond);

%% Plotting

% less data points
skipdata = 1600;

t      = linspace(0, simTime, simTime/simStepSize);
colors = lines(numRuns); % Get a colormap for different runs


%% define keep-out constraint
coneAngle = 20*pi/180;
b_B = [1;0;0];
n_I = [0;1;0];


% plot unit sphere and keep-out-cone
% figure(12)
% 
% plotConeOnUnitSphere(n_I, coneAngle,{'r'});
% h = zeros(1, 1);
% h(1) = plot(NaN,NaN,'r-');
% legend(h,'x-axis'); 



% cleanfigure();
% matlab2tikz('3D_scenario.tex','width','\figW','height','\figH');
save_sphere_view(50, 10, 'blub.pdf', x_sol_all, iter_all, [])
save_sphere_view(-43, 10, 'blub2.pdf', x_sol_all, iter_all, [])
%% plot unit sphere and keep-out-cone
fig = figure(10)
plotConeOnUnitSphere_mesh(n_I, coneAngle,{'r'});
hold on
for jj = 1:length(x_sol_all)

    
% extract the j-th run
x_sol_vec = x_sol_all{jj};
x_sol_vec = x_sol_vec(:,1:iter_all{jj});

% plot initial and final attitude attitude
b_I0 = mrp2trafo_BI(x_sol_vec(4:6,1))'*b_B;
b_IT = mrp2trafo_BI(x_sol_vec(4:6,end))'*b_B;
plot3(b_I0(1),b_I0(2),b_I0(3),'*','Color', 'g','LineWidth',1.5)
plot3(b_IT(1),b_IT(2),b_IT(3),'b*')

hold on
b_IT = zeros(3,length(1:skipdata:length(x_sol_vec)));

% transform instrument boresight inertial frame
idx = 1:skipdata:length(x_sol_vec); 
for j = 1:length(idx)

    b_IT(:,j) = mrp2trafo_BI(x_sol_vec(4:6,idx(j)))'*b_B;
end

% plot trajectory
plot3(b_IT(1,:),b_IT(2,:),b_IT(3,:),'Color', 'g')

end
hold on


% h = zeros(6, 1);
% h(1) = plot(NaN,NaN,'r-');
% h(2) = plot(NaN,NaN,'g-');
% h(3) = plot(NaN,NaN,'b-');
% h(4) = plot(NaN,NaN,'*g');
% h(5) = plot(NaN,NaN,'g-');
% h(6) = plot(NaN,NaN,'*b');
% legend(h,'x-axis','y-axis','z-axis','${_I}b_0$','${_I}b(t)$','origin'); 
% 
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10);
% xlabel('$x_I$', 'Interpreter', 'latex', 'FontSize', 10);      % x-axis label
% ylabel('$y_I$', 'Interpreter', 'latex', 'FontSize', 10);      % y-axis label
% zlabel('$z_I$', 'Interpreter', 'latex', 'FontSize', 10);      % if 3D plot
axis off
% cleanfigure();
% matlab2tikz('3D_compII.tex','width','\figW','height','\figH');

% Optional: Tighten layout to avoid white space (especially in axes)
axis tight;                     % Tight axis limits
set(gca, 'LooseInset', [0 0 0 0]);  % Remove extra padding around axes

% Export high-res cropped raster PDF
exportgraphics(gca, 'myfigure_raster2.pdf', ...
    'ContentType', 'image', ...       % Export as raster (not vector)
       'BackgroundColor', 'none', ...
    'Resolution', 600);               % Adjust resolution (300–600 is typical)
