%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Safe-by-Design: Approximate Nonlinear Model 
%   Predictive Control with Realtime Feasibility" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: Compute a approx. constraint set in six indeterminate (rates and 
%                      Modified Rodrigues Parameter (MRP)) for one keep-ot cone, 
%					   rate constraints and a constraint to bound the length 
%					   of the MRP. A sampling based approach is used in the end to 
%                      check the solution quality on the unit-sphere. 
%
%
%   License: see License file in repository.   
%
%   Needed software: - CasADi 3.6 
%					 - CaSoS
%
%
% ------------------------------------------------------------------------



clc
clear
close all

import casos.toolboxes.sosopt.*

% define keep-out constraint
coneAngle = 20*pi/180;

% boresight and undesired direction may not be the same; Think about 
% n_I * T_IB * b_B where T_IB is the identity matrix. In this case our
% boresight would lie in the keep out zone which is not possible
b_B = [1;0;0];
n_I = [0;1;0];


addpath helperFunctions\

% g(x) <= 0
[keepOut_con_casos,keepOut_con_sym] = generate_KeepOut_constraint2(b_B,n_I,coneAngle);

x  = casos.Indeterminates('x',6);

% simple bounds on rates
omegaMax1 = 0.5*pi/180;
omegaMax2 = 0.2*pi/180;
omegaMax3 = 0.2*pi/180;


x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';

n = 4;
g0 = (x(1)^2/omegaMax1^2)^(n/2) + (x(2)^2/omegaMax2^2)^(n/2) + (x(3)^2/omegaMax3^2)^(n/2) -1 ;

Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3)),0.5,0.5,0.5]);


% small term to ensure strict inequliaty
epsilon    = 1e-6*(x(4:6)'*x(4:6));

g = keepOut_con_casos - epsilon ; 

% rate constraints, keep-out cone and restriction to sublevel set
g = [g0;
     g;
	 x(4:6)'*x(4:6)-6];

% rescale
g = subs(g,x,inv(Dx)*x);

% SOS multiplier
if ~isempty(g) 
    s = casos.PS.sym('s',monomials(x,0:1),[length(g) 1],'gram');
else
    s = [];
end

% multiplier for monotonicty growth
s0 = casos.PS.sym('s0',monomials(x,0:1),'gram');

% allowable set function
h = casos.PS.sym('h',monomials(x,2:4));
h_sym = casos.PS.sym('h_sym',sparsity(h));


% initial guess for Coordinate-descent
h_star = (inv(Dx)*x)'*eye(6)*10000*(inv(Dx)*x);
b = 0.01;


%% Compute inner-estimate

disp('=========================================================')
disp('Build solver...')
tic

% define SOS feasibility to compute multiplier
sos = struct('x',s, ...
             'g',s*(h_sym-b) - g, ...
             'p',h_sym);

% states + constraint are SOS cones
opts.Kx.sos = length(g); 
opts.Kx.lin = 0; 
opts.Kc.sos = length(g);

% ignore infeasibility
opts.error_on_fail = false;

% solve by relaxation to SDP
S1 = casos.sossol('S1','mosek',sos,opts);

if ~isempty(g)
    s_sym = casos.PS.sym('s_sym',sparsity(s(1)),[length(g) 1]);
else
    s_sym = [];
end

% define SOS feasibility
sos2 = struct('x',[h;s0], ...
              'g',[s_sym*(h-b) - g;
                  s0*(h_sym-b) + b - h], ...	% "growth" constraint
              'p',[s_sym;h_sym]);

% states + constraint are SOS cones
opts.Kx.lin = 1; 
opts.Kx.sos = 1;
opts.Kc.sos = 1+length(g);

% ignore infeasibility
opts.error_on_fail = false;

% solve by relaxation to SDP
S2 = casos.sossol('S2','mosek',sos2,opts);
tbuild = toc;


disp('Finished building solver!')
disp('=========================================================')
disp('Start iteration...')

itermax = 100;

for iter = 1:itermax

   %% solve s-step
   sol1 = S1('p',h_star);
    
   % check solution
    switch (S1.stats.UNIFIED_RETURN_STATUS)
        case 'SOLVER_RET_SUCCESS'
              disp(['s step feasible in ' num2str(iter) '/' num2str(itermax) ] )
     
        otherwise
            disp(['s step infeasible in ' num2str(iter) '/' num2str(itermax) ] )
            break
    end
    
    %% solve h-step
    sol2 = S2('p',[sol1.x;h_star]);
    
    % check solution
    switch (S2.stats.UNIFIED_RETURN_STATUS)
        case 'SOLVER_RET_SUCCESS'
             h_star = sol2.x(1);
          
              disp(['h step feasible in ' num2str(iter) '/' num2str(itermax) ] )
        case {'SOLVER_RET_INFEASIBLE' 'SOLVER_RET_NAN'}
            disp(['h step infeasible in ' num2str(iter) '/' num2str(itermax) ] )
            break
        otherwise
            disp(['h step infeasible in ' num2str(iter) '/' num2str(itermax) ] )
            break
    end
end
tIter = toc-tbuild;

disp('=========================================================')
disp('Finished iteration')
disp(['Build time: ' num2str(tbuild) ' s' ])
disp(['Iteration time: ' num2str(tIter) ' s' ])
disp('_________________________________________________________')
disp(['Total time: ' num2str(tIter+tbuild) ' s' ])

% re-scale/scale input; remove very small term
h_res = cleanpoly(subs(h_star,x,Dx*x),1e-10)-b;

%% plotting

% define keep-out constraint
plotConeOnUnitSphere(n_I, coneAngle,{'r'});

% generate random attitude in MRP space with rate equal to zero
a = -1;
b =  1;

sigma = a +(b-a)*rand(3,10000);


% get only attitude that lie in safe set
hfun         = to_function(h_res);
hfun_values  = full(hfun(0,0,0,sigma(1,:),sigma(2,:),sigma(3,:)));

% find all samples that lie in the sublevel set
idx = find(hfun_values <= 0);

% plot safe attitudes in unit-sphere space i.e. we want that the borsight
% angle stays outside of the keep-out cones
for k = 1:length(idx)
    T_IB = mrp2trafo_BI(sigma(:,idx(k)))';
    borSight_I = T_IB*b_B;
    plot3(borSight_I(1),borSight_I(2),borSight_I(3),'*k')
end



% plot in MRP or rate domain
import casos.toolboxes.sosopt.*
figure(100)
pcontour(subs(subs(h_res,x(1:4),0),x,x),0,[-5 5 -5 5]./2,'k--')
hold on
pcontour(subs(subs(keepOut_con_casos,x(1:4),0),x,x),0,[-5 5 -5 5]./2,'r')
% pcontour(subs(subs(keepOut_con_casos1,x(3),0),x,x),0,[-5 5 -5 5]./2,'r')


figure(200)
pcontour3(subs(h_res,x(1:3),0),0,[-2 2 -2 2 -2 2]*1.2)
hold on
plot3(sigma(1,idx),sigma(2,idx),sigma(3,idx),'*k')

