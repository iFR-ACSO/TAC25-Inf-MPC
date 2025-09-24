%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: Script to execute  simulations for the 
%                      polynomial control law. The problem is setup
%                      using CasADi. The pre-computed CLF and CBF and
%                      control law as well as the weights Q and R are 
%                      and spacecraft specific data (bounds, inertia tensor) 
%                      are loaded from a terminalIngredients.mat file. 
%                      Different simulation modes are possible (single, MC and
%                      comparison). For single and comparsion, the user
%                      must select initial values such that B(x) <= 0, i.e,
%                      we must start in the CBF sublevel set (safe set).
%                      A .mex file is generated for the simulation (RK45) 
%                      to improve simulation time. 
%
%                       We want to solve the following:
%                       
%                       u(x(t)) = K_poly(x(t))
%
%                       where K_poly is the polynomial control law form the
%                       synthesize step.
%
%
%   Needed software:   - A C/C++ compiler for .mex file generation (We use a
%                      MS 2022 C/C++ compiler) 
%
%
%   License: see License file in repository.   
%
% ------------------------------------------------------------------------

import casadi.*

clear
clear textprogressbar  % to clear persistent variables
close all
clc

% functions to convert e.g. from MRP to Euler angles
addpath('..\helperFunc\')

SimulationType = 'Comparison'; % 'MC': Monte-carlo with uniform distribution
                               % 'Comparison': Several Pre-selected inital conditions
                               % 'Single': Single run

% load pre-computed data
load terminalIngredients.mat


%% Setup satellite parameter and dynamics
x  = SX.sym('x',6,1);
u  = SX.sym('u',3,1);
p  = SX.sym('p',length(x),1);

Nx = length(x);
Nu = length(u);

% ODE solve/simulation
dt0         = 0.1;
simTime     = 5000; % maxTime allowed
simStepSize = dt0;

% satellite dynamics (rates and MRP kinematics) 
% (Intertia tensor J read in above from terminalIngredients.mat)

% cross-product matrix
cpm = @(x) [   0   -x(3)  x(2);
              x(3)   0   -x(1);
             -x(2)  x(1)    0 ];

% MRP dynamics
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

% dynamics
xdot =  [-J\cpm(x(1:3))*J*x(1:3) + J\u;
         1/4*B(x(4:6))*x(1:3)];


% torque bounds (read in above from terminalIngredients.mat) 
u_low = umin;
u_up  = umax;

%% load weights, terminal penalty and invariant set 
% (weights are used below in cost function;
%  read in above from terminalIngredients.mat)

x_1 = x(1);
x_2 = x(2);
x_3 = x(3);
x_4 = x(4);
x_5 = x(5);
x_6 = x(6);

% pre-computed terminal set and penalty and their derivatives
W =    W_fun(x_1, x_2, x_3, x_4, x_5, x_6);
V =    V_fun(x_1, x_2, x_3, x_4, x_5, x_6);
K =    K_fun(x_1, x_2, x_3, x_4, x_5, x_6);

% % terminal set just to check if we are in the terminal set
Wfun = Function('f',{x},{W});   


% polynomial control law from synthesis
Kfun = Function('f',{x},{K});  

%% fixed-step Runge-Kutta 4 integration for simulation
f = Function('f', {x, u}, {xdot});

rk = 4;
dt = dt0/rk;
x0 = SX.sym('x0',size(x));
uk = SX.sym('uk',size(u));

% pre-allocate
k1 = SX(length(x),rk);
k2 = SX(length(x),rk);
k3 = SX(length(x),rk);
k4 = SX(length(x),rk);
xk = [x0 SX(length(x),rk)];

% loop over subintervals
for j=1:rk
    k1(:,j) = f(xk(:,j), uk);
    k2(:,j) = f(xk(:,j) + dt*k1(:,j)/2, uk);
    k3(:,j) = f(xk(:,j) + dt*k2(:,j)/2, uk);
    k4(:,j) = f(xk(:,j) + dt*k3(:,j), uk);
    xk(:,j+1) = xk(:,j) + dt*(k1(:,j) + 2*k2(:,j) + 2*k3(:,j) + k4(:,j))/6;
end

xkp1 = Function('fk', {x0 uk}, {xk(:,end)}, {'x0' 'uk'}, {'xk'});

%% generate matlab mex for simulation (just for speed up in matlab)
fprintf('Pre-compile mex-files for simulation.\n')
f = Function('f',{x,u},{xkp1(x,u)});
C = CodeGenerator('sim_mex.c');
C.add(f);
opts = struct('mex', true);

f.generate('sim_mex.c',opts);

mex("sim_mex.c")

%% Simulation preparation

% currently only setup for rest-to-rest profile; can be changed also to
% consider initial rates
if strcmp(SimulationType,'MC')
    
    % get 100 initial attitude; rates set to zero because rest-to-rest
    a4 = -0.55;  b4 = 0.55;
    x0_low(4:6,:) = (b4-a4)*rand(3,100)+a4;
    
    % we only consider initial states that lie in the terminal set
    idx = full(Wfun(x0_low)) <= 0;

    % reduce to feasible initial conditions
    x0_low = x0_low(:,idx);

    numRuns  = 100;

elseif strcmp(SimulationType,'Comparison')
     
     % single axis rest-to-rest maneuver
     numRuns  = 3;
   
     theta0 = 0;
     psi0   = 0;
     
     % must lie in CBF sublevel set! (not checked here)
     phi0   = [110;90;75]*pi/180;
      
     % we assume rest to rest, thus rate zero
     x0_low = zeros(6,numRuns);
     for j = 1:length(phi0)
        x0_low(4:6,j)  = Euler1232MRP([phi0(j),theta0,psi0]);
     end

     % .mat file name
     matFileName = 'polyControl_comparison_3_runs.mat';

else
    % single run
    numRuns  = 1;       % Define number of Monte-Carlo runs

     % just a single fixed initial attitude
     phi0   = 110*pi/180;
     theta0 = 0;
     psi0   = 0;
    
     x0_low(4:6,1)  = Euler1232MRP([phi0,theta0,psi0]);

     % .mat file name
     matFileName = 'polyControl_comparison.mat';

end


startSim = tic;

% Pre-allocate storage for multiple runs
x_sol_all   = cell(numRuns, 1);
u_sol_all   = cell(numRuns, 1);
tEnd_all    = nan(numRuns, simTime/simStepSize - 1);
suffCon_all = nan(numRuns, simTime/simStepSize - 1);
iter_all    = cell(numRuns, 1);

%% Simulations
for j = 1:numRuns
    fprintf('Simulation Run: %d/%d\n', j, numRuns);
    
    % get initial conditions for the j-th run
    x0 = x0_low(:,j); 
    
    % pre-allocated arrays
    x_sol_vec_poly = zeros(6, simTime/simStepSize);
    u_sol_vec_poly = zeros(3, simTime/simStepSize - 1);
   
    % set initial conditions  
    x_sim                = x0;
    x_sol_vec_poly(:,1)  = x0;

    % MRP to Euler anlges
    [phi,theta,psi]       =  mrp2eul(x0(4:6));
    x_sol_vec_poly(4:6,1) = [phi,theta,psi]'*180/pi;

    % first initial guess for decision variables
    z0 = zeros(Nu,1);

    textprogressbar('Simulation for horizon-one MPC:');

    for k = 2:simTime/simStepSize

        % get control at current state
        startTic =  tic;
            u_sol = full(Kfun(x0));
        tEnd_all(j, k-1) = toc(startTic);

        % store for later plotting
        u_sol_vec_poly(:,k-1) = u_sol;
        
        % Simulate
        x_sim(:,k)      = sim_mex(x_sim(:,k-1), u_sol);
        
        % store solution for later plotting
        x_sol_vec_poly(:,k)  = x_sim(:,k);
        
        % store Euler Angles in degree for plotting and convergence check
        [phi,theta,psi]  =  mrp2eul(x_sim(4:6,k));
       
        x_sol_vec_poly(4:6,k) = [phi,theta,psi]'*180/pi;
        

        % check convergence 
        if norm(x_sol_vec_poly(1:3,k),inf)*180/pi < 1e-3 && ... % below 0.001 deg/s
           norm(x_sol_vec_poly(4:6,k),inf) < 0.3 &&...          % below 0.1 deg
           norm(u_sol,inf) < 1e-3                               % below 1 mNm

           break
        end

        % state for next step
        x0 = x_sim(:,k);

        % use current solution as initial guess for next iteration
        z0 = u_sol;         
        
        % update progress bar
        textprogressbar(k/(simTime/simStepSize)*100);
    end
    
    textprogressbar('Progress bar  - termination')

    % Store results for each run
    x_sol_all{j}      = x_sol_vec_poly;
    iter_all{j}       = k;
    u_sol_all{j}      = u_sol_vec_poly;
    
end

% total time for Monte-carlo
simTimeMeas = toc(startSim);
fprintf('\nTotal Simulation time: %f seconds\n', simTimeMeas);

%% Compute statistics on computation time in miliseconds
minSolveTime_poly = min(tEnd_all, [], 'all') * 1000;
maxSolveTime_poly = max(tEnd_all, [], 'all') * 1000;


% Convert matrix to cell array, one cell per row
tEnd_all_cell = mat2cell(tEnd_all, ones(1, size(tEnd_all, 1)), size(tEnd_all, 2));

% Remove NaNs and make each output a column vector
tEnd_all_cleaned = cellfun(@(row) row(~isnan(row))', tEnd_all_cell, 'UniformOutput', false);

% Concatenate all cleaned column vectors and compute the mean in ms
meanSolveTime_poly= mean(cell2mat(tEnd_all_cleaned)) * 1000;

fprintf('Minimum solve time: %f ms\n', minSolveTime_poly);
fprintf('Maximum solve time: %f ms\n', maxSolveTime_poly);
fprintf('Mean solve time: %f ms\n', meanSolveTime_poly);

%% Plotting
t = linspace(0, simTime, simTime/simStepSize);
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
        plot(t(1:iter_all{j}), x_sol_all{j}(i,1:iter_all{j}) * 180/pi, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\omega_%c [°/s]', 'x' + (i-1)));
    grid on;

    % plot gray shadded area and dashed gray line
    xLimits = [0, t(max([iter_all{:}])) ]; 
    yDashed = x_low(i); 
    plot([0 t(max([iter_all{:}]))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    miny =  x_low(i)+0.5* x_low(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(max([iter_all{:}]))]; 
    yDashed =  x_up(i);
    plot([0 t(max([iter_all{:}]))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    maxy = x_up(i)+0.5*x_up(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(max([iter_all{:}])) miny maxy])
end

% Plot Attitude in Euler-Angles in degree
figure('Name', 'Attitude');
Euler_names = {'\phi','\theta','\psi'};
for i = 4:6
    subplot(3,1,i-3);
    hold on;
    for j = 1:numRuns
        plot(t(1:iter_all{j}), x_sol_all{j}(i,1:iter_all{j}), 'Color', colors(j, :));
    end
    % axis([0 t(max([iter_all{:}])) -180 180])
    xlabel('t [s]');
    ylabel([Euler_names{i-3} ' [deg]']);
    grid on;
end


% Plot Control Torques in miliNetwonmeter
t_short = linspace(0, simTime, (simTime/simStepSize)-1);

% set up with torques in mili-Newtonmeter
ulow = u_low'*1000;
uup  = u_up'*1000;

figure('Name', 'Control Torques')
for i = 1:3
    subplot(3,1,i);
    hold on;
    for j = 1:numRuns
        % stored control torques also in mili-Newtonmeter
        plot(t(1:iter_all{j}-1), u_sol_all{j}(i,(1:iter_all{j}-1)) * 1000, 'Color', colors(j, :));
    end
    xlabel('t [s]');
    ylabel(sprintf('\\tau_%c [mNm]', 'x' + (i-1)));
    grid on;

    % plot gray shadded area and dashed gray line
    xLimits = [0, t(max(iter_all{j}))]; 
    yDashed = ulow(i); 
    miny =  ulow(i)+0.5* ulow(i);
    plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5])
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [miny miny yDashed yDashed], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');

    xLimits = [0, t(max(iter_all{j}))]; 
    yDashed =  uup(i); 
    plot([0 t(max(iter_all{j}))],[yDashed yDashed],'--','Color',[0.5 0.5 0.5]) 
    maxy = uup(i)+0.5*uup(i);
    fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [yDashed yDashed maxy maxy], [0.7 0.7 0.7],'FaceAlpha',0.4 ,'EdgeColor', 'none');
    axis([0 t(max(iter_all{j})) miny maxy])

end

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

% %% store data for comparison
Q_poly_infinite     = Q;
R_poly_infinite     = R;
iter_conv_poly_inf  = iter_all;
x_sol_vec_poly      = x_sol_all;
u_sol_vec_poly      = u_sol_all;

tEnd_all_poly       = tEnd_all;

Omega_bounds = [omegaMax1;omegaMax2;omegaMax3];

cd ..\
save(matFileName,'Q_poly_infinite','R_poly_infinite','tEnd_all_poly',...
      'x_sol_vec_poly','u_sol_vec_poly','u_low','u_up','iter_conv_poly_inf',...
       "maxSolveTime_poly",'meanSolveTime_poly')

cd poly_ControlLaw\

rmpath('..\helperFunc\')
