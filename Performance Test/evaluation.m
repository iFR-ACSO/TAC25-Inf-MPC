%% ------------------------------------------------------------------------
%
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: This script is used to evaluate the second test case
%   in [1]. Plots are generated, the performance is evaluated and the plots
%   (and if desired) the tikz pictures found in [1] are generated.
%
%   License: see License file in repository.   
%
%   References: [1] Olucak, J, de Oliveira, A.C.B. and Cunis, T - 
%   Infinitesimal-horizon model predictive control as control barrier and 
%   Lyapunov function approach, submitted to IEEE Transaction on Automatic
%   Control
%
% ------------------------------------------------------------------------

close all
clear
clc


addpath("helperFunctions\")

% load MC data
load 'full_WS_KeepOutResults_withCompFun.mat'



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

skipdata = 100;

t      = linspace(0, simTime, simTime/simStepSize);
colors = lines(numRuns); % Get a colormap for different runs

% re-scale rate constraints (real physical constraints)
x_low =  [-omegaMax1*180/pi -omegaMax2*180/pi -omegaMax3*180/pi]';
x_up  =  [ omegaMax1*180/pi  omegaMax2*180/pi  omegaMax3*180/pi]';

% Plot Rates in Degree/second
figure('Name', 'Rates');
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        plot(t(1:skipdata:iter_all{j}), x_sol_all{j}(i,1:skipdata:iter_all{j}) * 180/pi, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\omega_%c [°/s]', 'x' + (i-1)));
    grid on;
    h = zeros(1, 1);
    h(1) = plot(NaN,NaN,'k--');
    legend(h,'zero line')

    % plot gray shadded area and dashed gray lines
    xLimits = [0, t(max(max(cell2mat(iter_all)))) ]; 
    yDashed = x_low(i); 
    plot([0 t(max(max(cell2mat(iter_all))))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    miny =  x_low(i)+0.5* x_low(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(max(max(cell2mat(iter_all))))]; 
    yDashed =  x_up(i);
    plot([0 t(max(max(cell2mat(iter_all))))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    maxy = x_up(i)+0.5*x_up(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(max(max(cell2mat(iter_all)))) miny maxy])
end


cleanfigure();
matlab2tikz('rates_compII.tikz','width','\figW','height','\figH');


% Plot Attitude in Euler-Angles in degree
figure('Name', 'Attitude');
Euler_names = {'\sigma_1','\sigma_2','\sigma_3'};
for i = 4:6
    subplot(3,1,i-3);
    hold on;
    for j = 1:numRuns
        plot(t(1:skipdata:iter_all{j}), x_sol_all{j}(i,1:skipdata:iter_all{j}), 'Color', colors(j, :));
    end
    axis([0 t(max(max(cell2mat(iter_all)))) -1 1])
    xlabel('t [s]');
    ylabel([Euler_names{i-3} ' [-]']);
    grid on;
    h = zeros(1, 1);
    h(1) = plot(NaN,NaN,'k--');
    legend(h,'zero line')
end


cleanfigure();
matlab2tikz('attitude_compII.tex','width','\figW','height','\figH');

% Plot Control Torques in miliNetwonmeter
t_short = linspace(0, simTime, (simTime/simStepSize)-1);

% set up with torques in miliNetwonmeter
ulow = u_low'*1000;
uup  = u_up'*1000;

figure('Name', 'Control Torques')
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        % Torques in miliNetwonmeter
        plot(t(1:skipdata:iter_all{j}-1), u_sol_all{j}(i,(1:skipdata:iter_all{j}-1)) * 1000, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\tau_%c [mNm]', 'x' + (i-1)));
    grid on;
    h = zeros(1, 1);
    h(1) = plot(NaN,NaN,'k--');
    legend(h,'zero line')
    % plot gray shadded area and dashed gray line
    xLimits = [0, t(max(max(cell2mat(iter_all))))]; 
    yDashed = ulow(i); 
    miny =  ulow(i)+0.5* ulow(i);
    plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(max(max(cell2mat(iter_all))))]; 
    yDashed =  uup(i); 
    plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5]) 
    maxy = uup(i)+0.5*uup(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(max(iter_all{j})) miny maxy])

end
h = zeros(1, 1);
h(1) = plot(NaN,NaN,'k--');
legend(h,'zero line')

cleanfigure();
matlab2tikz('torques_compII.tex','width','\figW','height','\figH');



% Plot Sufficient Conditions evaluted along trajectories
figure('Name', 'Sufficient Conditions');

axis([0 t(max(max(cell2mat(iter_all)))) minSuffCond 0.01])
hold on;
for j = 1:numRuns
    plot(t(1:skipdata:iter_all{j}-1), suffCon_all(j,(1:skipdata:iter_all{j}-1)), 'Color',colors(j, :));
end
plot([0 t(max(max(cell2mat(iter_all))))], [0 0.0], 'k--');
xlabel('t [s]');
ylabel('\nabla \hat V \cdot f(x,u) + L(x,u)');

h = zeros(1, 1);
h(1) = plot(NaN,NaN,'k--');
legend(h,'zero line')

cleanfigure();
matlab2tikz('suffCon_compII.tex','width','\figW','height','\figH');


% Plot barrier evaluted along trajectories
figure('Name', 'Barrier');

hold on;
for j = 1:numRuns
       plot(t(1:skipdata:iter_all{j}), Barrier_all(j,(1:skipdata:iter_all{j})), 'Color',colors(j, :));
end
xlabel('t [s]');
ylabel('\hat h(x)');
plot([0 t(max(max(cell2mat(iter_all))))], [0 0.0], 'k--');
axis([0 t(max(max(cell2mat(iter_all)))) -0.011 0.001])

h = zeros(1, 1);
h(1) = plot(NaN,NaN,'k--');
legend(h,'zero line')
cleanfigure();
matlab2tikz('barrier_compII.tex','width','\figW','height','\figH');

% Plot Solve Time
figure('Name', 'Solve Time');

for j = 1:numRuns
    semilogy(t_short(1:iter_all{j}-1), tEnd_all(j,(1:iter_all{j}-1)), 'Color', colors(j, :));
    hold on;
end
xlabel('Simulation time [s]');
ylabel('Computation time [s]');
axis([0 t(max(iter_all{j})) 1e-6 1])
grid on;


%% define keep-out constraint
coneAngle = 20*pi/180;
b_B = [1;0;0];
n_I = [0;1;0];

% plot unit sphere and keep-out-cone
figure(10)
% subplot(1,3,1)
% plotConeOnUnitSphere_slight(n_I, coneAngle,{'r'});
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10);
% xlabel('$x_I$', 'Interpreter', 'latex', 'FontSize', 10);      % x-axis label
% ylabel('$y_I$', 'Interpreter', 'latex', 'FontSize', 10);      % y-axis label
% zlabel('$z_I$', 'Interpreter', 'latex', 'FontSize', 10);      % if 3D plot


subplot(1,2,1);
plotConeOnUnitSphere_mesh(n_I, coneAngle,{'r'});
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10);
xlabel('$x_I$', 'Interpreter', 'latex', 'FontSize', 10);      % x-axis label
ylabel('$y_I$', 'Interpreter', 'latex', 'FontSize', 10);      % y-axis label
zlabel('$z_I$', 'Interpreter', 'latex', 'FontSize', 10);      % if 3D plot
hold on
for jj = 1:length(x_sol_all)

% extract the j-th run
x_sol_vec = x_sol_all{jj};

% plot initial and final attitude attitude
b_I0 = mrp2trafo_BI(x_sol_vec(4:6,1))'*b_B;
b_IT = mrp2trafo_BI(x_sol_vec(4:6,end))'*b_B;
plot3(b_I0(1),b_I0(2),b_I0(3),'g*')
plot3(b_IT(1),b_IT(2),b_IT(3),'b*')

hold on
% transform instrument boresight inertial frame
for j = 2:length(x_sol_vec)-1
    b_It(:,j) = mrp2trafo_BI(x_sol_vec(4:6,j))'*b_B;
end

% plot trajectory
plot3(b_It(1,:),b_It(2,:),b_It(3,:),'g-')

end
ax2 = gca;
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10);
xlabel('$x_I$', 'Interpreter', 'latex', 'FontSize', 10);      % x-axis label
ylabel('$y_I$', 'Interpreter', 'latex', 'FontSize', 10);      % y-axis label
zlabel('$z_I$', 'Interpreter', 'latex', 'FontSize', 10);      % if 3D plot
h = zeros(6, 1);
h(1) = plot(NaN,NaN,'r-');
h(2) = plot(NaN,NaN,'g-');
h(3) = plot(NaN,NaN,'b-');
h(4) = plot(NaN,NaN,'*g');
h(5) = plot(NaN,NaN,'g-');
h(6) = plot(NaN,NaN,'*b');
legend(h,'x-axis','y-axis','z-axis','${_I}b_0$','${_I}b(t)$','origin', ...
    'Interpreter', 'latex', 'FontSize', 10,'Location','northoutside','NumColumns',6); 


subplot(1,2,2)
ax3 = gca;
tmpPos = ax3.Position;
delete(ax3);   

% Copy the second axes into the third subplot location
new_ax = copyobj(ax2, gcf);
set(new_ax, 'Position', tmpPos); 

% Change the view
view(new_ax, [45 30]); % Example: new view angle
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10);
xlabel('$x_I$', 'Interpreter', 'latex', 'FontSize', 10);      % x-axis label
ylabel('$y_I$', 'Interpreter', 'latex', 'FontSize', 10);      % y-axis label
zlabel('$z_I$', 'Interpreter', 'latex', 'FontSize', 10);      % if 3D plot

exportgraphics(gcf, 'scenario2.pdf', 'ContentType', 'vector');