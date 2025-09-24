%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: Script to find a suitable terminal set and penalty
%   for the full horizon NMPC formulation(s). SOS programming is used to
%   find the largest possible (based on bisection) stable level set for the
%   terminal set. The nonlinear system is linearized and a Lyapunov
%   function for the linear system is found by solving the Riccati
%   equation, i.e., by computing an LQR controller. Additionally, state and
%   control constraint are considered.
%
%   License: see License file in repository.   
%
%   Needed software: - CasADi 3.6 (for CaSoS)
%                    - CaSoS
%
%
% ------------------------------------------------------------------------


clc
clear
import casos.toolboxes.sosopt.cleanpoly
import casos.toolboxes.sosopt.pcontour

% system states
x = casos.Indeterminates('x',6);
u = casos.Indeterminates('u',3);

%% system dynamics

% cross-product matrix
cpm = @(x) [   0   -x(3)  x(2);
              x(3)   0   -x(1);
             -x(2)  x(1)    0 ];

B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

% satellite parameter
I= diag([31046;77217;78754]);

% dynamics

f =  [-inv(I)*cpm(x(1:3))*I*x(1:3) + inv(I)*u;
         1/4*B(x(4:6))*x(1:3)];


% compute Lyapunov for linear system
A0 = full(subs(nabla(f,x),[x;u],zeros(9,1)));
B0 = full(subs(nabla(f,u),[x;u],zeros(9,1)));

% same weights as 
Q = diag([1,1,1,1,1,1]);
R = eye(3);

[K,P] = lqr(A0,B0,Q,R);

% closed-loop dynamics
f = subs(f,u,-K*x);


% simple bounds on rates;
omegaMax1 = 0.5*pi/180;
omegaMax2 = 0.2*pi/180;
omegaMax3 = 0.2*pi/180;

% bounds and scaling
x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';

Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3)),0.5,0.5,0.5]);

n = 2;
g0 = (x(1)^2/omegaMax1^2)^(n/2) + (x(2)^2/omegaMax2^2)^(n/2) + (x(3)^2/omegaMax3^2)^(n/2) + ...
     (x(4)^2/0.57^2)^(n/2) + (x(5)^2/0.57^2)^(n/2) + (x(6)^2/0.57^2)^(n/2) - 1;

% re-scale input of state constraints
g0 = subs(g0,x,inv(Dx)*x); 

% remove very small terms
f = cleanpoly(Dx*subs(f,x,inv(Dx)*x),1e-10);

% Lyapunov function candidate of linear system
V = (inv(Dx)*x)'*P*(inv(Dx)*x);

% derivative w.r.t. time
Vdot = nabla(V,x)*f;


% SOS multiplier
s  = casos.PS.sym('q',monomials(x,0:2),'gram');
s1 = casos.PS.sym('q1',monomials(x,0:1),[3,1],'gram');
s2 = casos.PS.sym('q2',monomials(x,0:1),[3,1],'gram');
s3 = casos.PS.sym('q3',monomials(x,0:1),'gram');

umin  = -ones(3,1)*1.2;
umax  =  ones(3,1)*1.2;

% enforce positivity
l = 1e-6*(x'*x);
% level of stability
g = casos.PS.sym('g');

%% Bisection

% define SOS feasibility problem
sos = struct('x',[s;s1;s2;s3], ...
             'g',[s*(V-g)-Vdot-l; ...               
                  s1*(V-g) + (-K*inv(Dx)*x) - umin; ...
                  s2*(V-g) + umax - (-K*inv(Dx)*x); ...
                  s3*(V-g) - g0 ...
                  ], ...
             'p',g);

% states + constraint are SOS cones
opts.Kx = struct('sos', 8);
opts.Kc = struct('sos', 8);

% ignore infeasibility
opts.error_on_fail = false;
tic
% solve by relaxation to SDP
S = casos.sossol('S','mosek',sos,opts);

% find largest stable level set
lb = 0; ub = 100;

% bisection to find largest possible stable level set
while (ub - lb > 1e-4)
    ptry = (lb+ub)/2;

    % evaluate parametrized SOS problem
    sol = S('p',ptry);

    switch (S.stats.UNIFIED_RETURN_STATUS)
        case 'SOLVER_RET_SUCCESS', lb = ptry;
               fprintf(' g is feasible: %g.\n', ptry)
        otherwise
            ub = ptry;
            fprintf(' g is infeasible: %g.\n', ptry)

    end
end
toc
fprintf('Maximal stable level set is %g.\n', lb)


% Terminal set and cost 
X_T = x'*P*x - lb;
V_T =  x'*P*x;

%%
import casos.toolboxes.sosopt.pcontour
figure()
pcontour(subs(X_T,x(1:4),zeros(4,1)),0,[-1 1 -1 1])

