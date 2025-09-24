%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: Script to execute simulations for the  
%                      full horizon NMPC using a custom RTI scheme. 
%                      The problem is setup using CasADi. We make use of 
%                      CasADi's nlpsol interface. Different simulation 
%                      modes are possible (single and comparison). We make 
%                      use of a multiple-shooting formulation.
%
%                       We want to solve the following NLP:
%
%                       min_{u_k ..., u_{N-1},x_k ,... x_N}  V + \sum x^T Q x + u^T R u 
%                       s.t.   x_{k+1} = f(x,u) 
%                              x \in X
%                              x_N \in X_N
%                              u \in U
% 
%                       where Q, R are pre-defined, V and W are
%                       pre-computed terminal penalty and invariant set
%                       respectively. X is the constrained set (rate
%                       constraints and MRP constraint). X_N is the
%                       terminal set.
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


close all
clear
clear textprogressbar
clc

addpath('..\helperFunc\')

import casadi.*


solvermethod = 'RTI'; 


SimulationType = 'Comparison'; % 'Comparison': Several Pre-selected inital conditions
                               % 'Single': Single run

%% Simulation parameter
simTime = 5000;
simStepSize = 0.1;


%% Simulation Type and initial conditions
if strcmp(SimulationType,'Single')
    % single run
    numRuns  = 1;       % Define number of Monte-Carlo runs

     % just a single fixed initial attitude
     phi0   = 75*pi/180;
     theta0 = 0;
     psi0   = 0;
    
     x0_low(4:6,1)  = Euler1232MRP([phi0,theta0,psi0]);

    
     % define integration step size for different initial conditions
     h_vec = 2;
     T_vec = 200;         % prediction horizon
     
else
     numRuns  = 3;

     theta0 = 0;
     psi0   = 0;

     phi0   = [110;90;75]*pi/180;

     % we assume rest to rest, thus rate zero
     x0_low = zeros(6,numRuns);

     for j = 1:length(phi0)
        x0_low(4:6,j)  = Euler1232MRP([phi0(j),theta0,psi0]);
     end

    % define integration step size for different initial conditions
    h_vec = [4; 3; 2];
    T_vec = [400; 300; 200];         % prediction horizon (heuristically found!)

    
end


startSim = tic;

% Pre-allocate storage for multiple runs
x_sol_all   = cell(numRuns, 1);
u_sol_all   = cell(numRuns, 1);
tEnd_all    = nan(numRuns, simTime/simStepSize - 1);
iter_all    = cell(numRuns, 1);


% We re-setup the NMPC depeing on the initial condition, i.e., we adjust
% the prediction horizon and hence no. of decision variables and
% constraints
for j = 1:numRuns


%% define cost function
R = diag([1;1;1]);
Q = diag([1,1,1,1,1,1]);

%% define problem
% variables
x  = SX.sym('x',6,1);
u  = SX.sym('u',3,1);


nx = length(x);
nu = length(u);

% adjust prediction horizon based on select prediction time and step-length
N = T_vec(j)/h_vec(j);     


% cross-product matrix
cpm = @(x) [   0   -x(3)  x(2);
              x(3)   0   -x(1);
             -x(2)  x(1)    0 ];

% MRP 
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

% satellite parameter
I = diag([31046;77217;78754]);

% dynamics
xdot =  [-I\cpm(x(1:3))*I*x(1:3) + I\u;
         1/4*B(x(4:6))*x(1:3)];

% compute terminal set and terminal penalty
A = full(casadi.DM(substitute(jacobian(xdot,x),[x;u],[zeros(6,1);zeros(3,1)])));
B = full(casadi.DM(substitute(jacobian(xdot,u),[x;u],[zeros(6,1);zeros(3,1)])));

[K,P] = lqr(A,B,Q,R);

% level set; found with SOS program i.e. terminalIngredient_full.m
gamma = 60.1815;

% terminal set was computed via constrained ROA; here penalty
Phi = Function('f',...      % Name
            {x, u,},...     % Input variables
            {x'*P*x});              

% path constraints ( here: sigma'*simga <= 1=
h0     = x(4:6)'*x(4:6); 
h0_low = 0;
h0_up  = 1; 

% simple bounds; MRP parameter are restricted with path constraint
x_low    =  [-0.5*pi/180 -0.2*pi/180 -0.2*pi/180 -inf -inf -inf]';
x_up     = -x_low';

maxu = 1.2; % Nm
u_low = [-maxu -maxu -maxu]';
u_up  = [ maxu maxu  maxu]';


Nx = length(x);
Nu = length(u);

tic
disp('-----------------------------------')
disp('Full hor. problem')

p  = SX.sym('p',length(x),1);

Q_weight = Q;
R_weight = R;

% path constraint function
H = Function('f',...       % Name
             {x,p},...     % Input variables
             {h0});   


%% fixed-step Runge-Kutta 4 integration; compare to casadi example package

f = Function('f', {x, u}, {xdot}); 

rk = 4; % subintervals
dt = T_vec(j)/N/rk;
x0 = SX.sym('x0',size(x));
uk = SX.sym('uk',size(u));

k1 = SX(length(x),rk); 
k2 = SX(length(x),rk); 
k3 = SX(length(x),rk); 
k4 = SX(length(x),rk);
xk = [x0 SX(length(x),rk)];

for jj=1:rk
    k1(:,jj) = f(xk(:,jj), uk);
    k2(:,jj) = f(xk(:,jj) + dt*k1(:,jj)/2, uk);
    k3(:,jj) = f(xk(:,jj) + dt*k2(:,jj)/2, uk);
    k4(:,jj) = f(xk(:,jj) + dt*k3(:,jj), uk);
    xk(:,jj+1) = xk(:,jj) + dt*(k1(:,jj) + 2*k2(:,jj) + 2*k3(:,jj) + k4(:,jj))/6;
end
xkp1 = Function('fk', {x0 uk}, {xk(:,end)}, {'x0' 'uk'}, {'xk'});


%% decision variables
X = SX.sym('X', [length(x) N]);
U = SX.sym('U', [length(u) N]);

% vector of decision variables
z = [X(:) ;U(:)];

% simple constraints on control
control_lb_grid = repmat(u_low,1,N);
control_ub_grid = repmat(u_up,1,N);

% simple constraints on states
state_lb_grid = repmat(x_low,1,N);
state_ub_grid = repmat(x_up,1,N);

lbz = [state_lb_grid(:); control_lb_grid(:)];
ubz = [state_ub_grid(:); control_ub_grid(:)];

%% path constraints
% dynamics
g = X - xkp1([x0 X(:,1:N-1)], U); %  multiple shooting
g = reshape(g,1,size(g,1)*size(g,2));

% equality constraint for dynamics
lbg_dyn = zeros(1,size(g,2));
ubg_dyn = lbg_dyn;

% add path constraints and terminal constraint
gh = [ H(X(:,1:end),p), gamma-X(:,end)'*P*X(:,end)] ; 
gH = reshape(gh,1,size(gh,1)*size(gh,2));

% put all path constraints together
g = [g,gH];

lbg_cust = repmat(h0_low ,1,N);
ubg_cust = repmat(h0_up ,1,N);

% combine path constraints (defect constraints and user defined
% in-/equality constraints
lbg = [lbg_dyn,lbg_cust,0 ] ;
ubg = [ubg_dyn,ubg_cust,inf ];

%% cost function
Q = Function('f', {x, u}, { x'*Q_weight*x + u'*R_weight*u});


J = Phi(X(:,end),zeros(size(u))) + sum(Q([x0 X(:,1:end-1)],U(:,1:end))); 

%% initial guess for first iteration
z0 = zeros(N*(Nx+Nu),1);

%% setup solver
prob   = struct('f', J, 'x', z, 'g', g,'p',x0);

fprintf('Setup nlp. solver using %s.\n',solvermethod)

options = struct('print_status',0,...
                     'print_header',0,...
                     'print_time',0,...
                     'record_time',true,...
                     'verbose_init',0,...
                     'print_out',false,...
                     'print_iteration',false,...
                     'max_iter',1);  % Maximum number of SQP iterations (RTI)

% supress all display output of underlying QP solver (qrqp specific)
options.qpsol_options.print_info   = false;
options.qpsol_options.print_out    = false;
options.qpsol_options.print_in     = false;
options.qpsol_options.print_header = false;
options.qpsol_options.print_iter   = false;
 
options.qpsol_options.error_on_fail = false;
options.qpsol = 'qrqp';

solver = casadi.nlpsol('solver', 'sqpmethod', prob, options);

fprintf('Solver is setup using %s.\n',solvermethod)


%% prepare for next run
fprintf('Simulation Run: %d/%d\n', j, numRuns);
    
% get initial conditions for the j-th run
x0 = x0_low(:,j); 
    
% pre-allocated arrays for j-th run
x_sol_vec = nan(6, simTime/simStepSize);
u_sol_vec = nan(3, simTime/simStepSize - 1);
 
simSteps = simTime/simStepSize; 

x_sim(:,1)       = x0_low(:,j);
[phi,theta,psi]  = mrp2eul(x_sim(4:6,1));

% store first step
x_sol_vec(1:3,1) = x0_low(1:3,1);
x_sol_vec(4:6,1) = [phi,theta,psi]'*180/pi;


textprogressbar('Simulation for full-horizon MPC:');


%% REMARK:
%           Actually we also provide lagrange multiplier as initial guess.
%           However, this resulted in bad behavior of the RTI scheme and is
%           hence neglected. Uncomment the lines below to run with lagrange
%           multiplier.
%


%%
% initial guess for langrange multiplier
% lam_g0  = zeros(length(lbg),1);
% lam_x0  = zeros(length(z0),1);

for k = 2:simSteps
        
        % solve full horizon NMPC problem
        % if k > 2
        % [sol]   = solver('x0',  z0,'p', x_sim(:,k-1),'lbx', lbz,'ubx', ubz,'lbg', lbg,'ubg', ubg,'lam_x0',lam_x0,'lam_g0',lam_g0);
        % else
            [sol]   = solver('x0',  z0,'p', x_sim(:,k-1),'lbx', lbz,'ubx', ubz,'lbg', lbg,'ubg', ubg);
        % end
        
        % get wall time via CasADi interface
        tEnd_all(j, k-1) = solver.stats.t_wall_total ;     
        
        % get solution
        z_opt = full(sol.x);
        
        % prediction
        x_sol = reshape(z_opt(1:Nx*N),Nx,N);
        
        % control 
        u_sol = z_opt((nx*N)+1:end);
        u_sol = reshape(u_sol,3,N);
        
        % only apply first control to plant
        u_sol_vec(:,k-1) = u_sol(:,1);
        
        % apply first entry of optimal solution
        x_sim(:,k) = sim_mex(x_sim(:,k-1), u_sol(:,1));

        % store Euler Angles in degree for plotting and convergence check
        [phi,theta,psi]  =  mrp2eul(x_sim(4:6,k));
        x_sol_vec(1:3,k) = x_sim(1:3,k);
        x_sol_vec(4:6,k) = [phi,theta,psi]'*180/pi;
        

        % check convergence 
        if norm(x_sim(1:3,k),inf)*180/pi < 1e-3 && ...   % below 0.001 deg/s
           norm([phi,theta,psi],inf)*180/pi < 0.3 &&...  % below 0.3 deg
           norm(u_sol(:,1),inf) < 1e-3                   % below 1 mNm

           break
        end
        
        % initial guess
        z0 = z_opt;

        % lam_g0 = sol.lam_g;
        % lam_x0 = sol.lam_x;

        % just in case something goes wrong aboard simulation!
        if strcmp(solver.stats.unified_return_status,'SOLVER_RET_NAN')
            disp('Problem infeasible!')
            break
        end

        % update progress bar
        textprogressbar(k/(simTime/simStepSize)*100);
end 


    % Store results for j-th run
    x_sol_all{j}      = x_sol_vec;
    iter_all{j}       = k;
    u_sol_all{j}      = u_sol_vec;

    textprogressbar('Progress bar  - termination')

end

totalSimTime = toc(startSim);

%% plotting

t = linspace(0, simTime, simTime/simStepSize);
colors = lines(numRuns); % Get a colormap for different runs

% re-scale rate constraints (real physical constraints)
x_low =  [-0.5 -0.2 -0.2]'; % in degree/second
x_up  =  -x_low;

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

    % plot gray shadded area and dashed gray lines
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
    xlabel('t [s]');
    ylabel([Euler_names{i-3} ' [deg]']);
    grid on;
end


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

%% store full workspace
save('full_horizon_MPC_simulation_completeWS_3_runs_RTI.mat')

rmpath('..\helperFunc\')
