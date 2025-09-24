%% ------------------------------------------------------------------------
%   
%   Supplementary Material for "Infinitesimal-horizon model predictive 
%   control as control barrier and Lyapunov function approach" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%   Short Description: Script to syntheszise the non-conflicting CBF/CLF for 
%                      for the infinetesimal-MPC approach for simple bound 
%                      constraints and one keep-out cone. The constraints
%                      are approximated in "innerApprox_allowSet.m", i.e.,
%                      an inner approximation of the allowable set is
%                      pre-computed.
%                      The CBF/CLF is pre-computed  and  stored in .mat file, 
%                      which is the used in   inf_MPC_simulation. 
%
%   License: see License file in repository.   
%
%   Needed software: - CasADi 3.6 
%                    - A C/C++ compiler for .mex file generation (We use a
%                      MS 2022 C/C++ compiler) 
%
%
% ------------------------------------------------------------------------

clear
close all
clc

% system states
x = casos.PS('x',6,1);
u = casos.PS('u',3,1);

%% Hubble telescope parameter
J = diag([31046;77217;78754]);

% simple bounds on rates;
omegaMax1 = 0.5*pi/180;
omegaMax2 = 0.2*pi/180;
omegaMax3 = 0.2*pi/180;

x_low =  [-omegaMax1 -omegaMax2 -omegaMax3]';
x_up  =  [ omegaMax1  omegaMax2  omegaMax3]';


% control constraint; assumption is that the box is inside the full
% torque volume. This is roughly estimated visually.
umin = [-1 -1 -1]'*1.2;
umax = [ 1  1  1]'*1.2;

Dx   = diag([1/(x_up(1)-x_low(1)),1/(x_up(2)-x_low(2)),1/(x_up(3)-x_low(3)),1,1,1]); 

Dxin = inv(Dx);

%% dynamics
% cross-product matrix
cpm = @(x) [   0  -x(3)  x(2); 
              x(3)   0   -x(1); 
             -x(2)  x(1)   0 ];

% dynamics
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

f =  [-J\cpm(x(1:3))*J*x(1:3) + J\u; % omega_dot
      1/4*B(x(4:6))*x(1:3)];         % sigma_dot

% trim point
x0    = [0 0 0 0 0 0]';
u0    = [0,0,0]';

A0 = full(casos.PD(subs(nabla(f,x),[x;u],[x0;u0])));
B0 = full(casos.PD(subs(nabla(f,u),[x;u],[x0;u0])));

% cost function weights
Q = diag([1, 1, 1, 1, 1 ,1]);
R = eye(3);

% generate an initial guess 
[K0,P0] = lqr(full(A0),full(B0),Q,R);

% scaled initial guess for terminal penalty (Lyapunov linear system)
Wval = (inv(Dx)*x)'*P0*(inv(Dx)*x);

% scale dynamics
f = Dx*subs(f,[x;u],[Dx\x;u]);


% state constraint (pre-computed using "innerApprox_allowSet.m")
g0 =    -0.01 - 19.2751*x(1)^2 - 171.929*x(2)^2 - 171.929*x(3)^2 + 0.00183104 ...
  *x(4)^2 + 0.0108846*x(4)*x(5) + 0.00183099*x(5)^2 + 0.0704132*x(6)^2  ...
  + 56.0706*x(1)^2*x(6) + 369.254*x(2)^2*x(6) + 369.269*x(3)^2*x(6) + 0.0341608 ...
  *x(4)^2*x(6) + 0.0669249*x(4)*x(5)*x(6) + 0.0341607*x(5)^2*x(6) + 0.0852434 ...
  *x(6)^3 + 2.08948e+06*x(1)^4 + 1.09443e+06*x(1)^2*x(2)^2 + 8.40133e+07 ...
  *x(2)^4 + 1.09441e+06*x(1)^2*x(3)^2 + 9.4321e+06*x(2)^2*x(3)^2 + 8.40134e+07 ...
  *x(3)^4 + 5.02581*x(1)^2*x(4)^2 + 28.3726*x(2)^2*x(4)^2 + 28.3714*x(3)^2 ...
  *x(4)^2 + 0.00400786*x(4)^4 - 37.1211*x(1)^2*x(4)*x(5) - 322.653*x(2)^2*x(4) ...
  *x(5) - 322.659*x(3)^2*x(4)*x(5) + 0.0151057*x(4)^3*x(5) + 5.02578*x(1)^2 ...
  *x(5)^2 + 28.3724*x(2)^2*x(5)^2 + 28.3712*x(3)^2*x(5)^2 + 0.0280411*x(4)^2 ...
  *x(5)^2 + 0.0151057*x(4)*x(5)^3 + 0.00400787*x(5)^4 - 18.2148*x(1)^2*x(6)^2  ...
  - 87.4636*x(2)^2*x(6)^2 - 87.5162*x(3)^2*x(6)^2 + 0.0232806*x(4)^2*x(6)^2  ...
  + 0.0523411*x(4)*x(5)*x(6)^2 + 0.0232806*x(5)^2*x(6)^2 + 0.0356876*x(6)^4;

% re-scale input of state constraints
g = subs(g0,x,Dx\x); 

%% setup SOS problem
load initGuess_termSet.mat

% terminal set (invariant set)
W  = casos.PS.sym('w',monomials(x,2:4));

% terminal penalty
V  = casos.PS.sym('v',monomials(x,2));


Kd = casos.PS.sym('kd',[3,1]);
Kp = casos.PS.sym('kp',[3,1]);

K = -Kd.*x(1:3)-Kp.*x(4:6);


% SOS mulitplier
s1 = casos.PS.sym('s1',monomials(x,2));
s2 = casos.PS.sym('s2',monomials(x,2));
s3 = casos.PS.sym('s3',monomials(x,0),[3 1]);
s4 = casos.PS.sym('s4',monomials(x,0),[3 1]);
s5 = casos.PS.sym('s5',monomials(x,4));

% fixed level set of CBF (same as constraint set)
b = 0.01;

% options for sequential sos
opts = struct('sossol','mosek');

opts.verbose       = 1;
opts.max_iter      = 100;


tau = ( nabla(V,x)*subs(f,u,K)    + K'*R*K + (inv(Dx)*x)'*Q*(inv(Dx)*x)); 

% find the "largest" possible state for rest-to-rest maneuver
cost = dot(subs(g,x(1:3),zeros(3,1))-subs(W,x(1:3),zeros(3,1)),subs(g,x(1:3),zeros(3,1))-subs(W,x(1:3),zeros(3,1)));

sos = struct('x', [W;V;Kd;Kp;s1;s2;s3; s4;s5],...  % decision variables
              'f', cost ,...                       % cost function
              'p',[]);                             % parameter


gammaW = 0.001;

% constraints
sos.('g') = [
             s1
             s2
             s3;
             s4;
             s5;
             V - 1e-6*(x'*x);                                     % CLF positivity              
             s1*(W-b) - g;                                        % State constraints
             s2*(W-b)  -  nabla(W,x)*subs(f,u,K) - gammaW*(W-b);  % CBF dissipation
             s3*(W-b)  + K-umin;                                  % control constraints
             s4*(W-b)  + umax-K;
             s5*(W-b) -  tau ;                                    % CLF dissipation
             ];

% states + constraint are linear/SOS cones
opts.Kx = struct('lin', length(sos.x));
opts.Kc = struct('sos', length(sos.g));
opts.sossol_options.newton_solver = [];
% solver setup
S  = casos.nlsossol('S','sequential',sos,opts);

% initial guess for sequential
x0 = casos.PD([ g; ...
                Wval;
                diag(K0(1:3,1:3)); ...         
                diag(K0(1:3,4:6)); ...
                ones(3,1);
                ones(3,1);
                x'*x; ...
                x'*x; ...
                x'*x]);
 
load initGuess_termSet.mat
x0 = casos.PD(monoGuess,coeffGuess);

% solve
sol = S('x0',x0);

bsol = b;

% re-scale invariant set, terminal penalty and local control law
Wsol_re = subs(sol.x(1),x,Dx*x) - full(casos.PD(bsol)); % store CBF as zero-sublevel set
Vsol_re = subs(sol.x(2),x,Dx*x);

K       = -sol.x(3:5).*x(1:3)-sol.x(6:8).*x(4:6);
Ksol_re = subs(K,x,Dx*x);



%% plotting
import casos.toolboxes.sosopt.*

% slice for rates
figure(1)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
clf
pcontour(subs(subs(Wsol_re,x(3:end),zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'g')
hold on 
pcontour(subs(subs(g0,x(3:end),zeros(4,1)),x,deg2rad*x),0,[-omegaMax1 omegaMax1 -omegaMax1 omegaMax1]*180/pi,'k--')
legend('Terminal Set','Safe Set')

% 3D slice for Modified rodrigues parameter
figure(2)
deg2rad = diag([pi/180,pi/180,pi/180 1 1 1]);
clf
pcontour3(subs(Wsol_re,x(1:3),zeros(3,1)),0,[-1 1 -1 1 -1 1])
hold on 
legend('Terminal Set')


%% verification 

% ------------------------------------------------------------------------
% Comment: Altough we have the guarantee all SOS constraints are met if 
% feasible, we can run a simple sampling approach to check if the 
% sufficient conditions are met.
% ------------------------------------------------------------------------

% unscaled dynamics
B = @(sigma) (1-sigma'*sigma)*eye(3)+ 2*cpm(sigma)+ 2*sigma*sigma';

f =  [-J\cpm(x(1:3))*J*x(1:3) + J\u;
      1/4*B(x(4:6))*x(1:3)];

% continous-time penalty, invariant set and control law as functions
penalty     = to_function(nabla(Vsol_re,x)*subs(f,u,Ksol_re) + Ksol_re'*R*Ksol_re + x'*Q*x) ; 
safety_fun  = to_function(Wsol_re);
Kfun        = to_function(Ksol_re);

% generate sample rate with the individual boubds
nSample = 1000000;          % number of samples
samples = zeros(6,nSample); % pre-allocation

a1 = -0.5*pi/180;
b1 =  0.5*pi/180;
samples(1,:) = (b1-a1)*rand(1,nSample)+a1;

a2 = -0.2*pi/180;
b2 =  0.2*pi/180;
samples(2,:) = (b2-a2)*rand(1,nSample)+a2;

a3 = -0.2*pi/180;
b3 =  0.2*pi/180;

samples(3,:) = (b3-a3)*rand(1,nSample)+a3;

a4 = -1;
b4 =  1;

samples(4:6,:) = (b4-a4)*rand(3,nSample)+a4;


% get all samples that lies within the invariant set, i.e.,  W(x_samp) <= 0
idx = find(full(safety_fun(samples(1,:),samples(2,:),samples(3,:),samples(4,:),samples(5,:),samples(6,:))) <= 0);

% evaluate control law and check if commanded control torques lie in bounds
uval = full(Kfun(samples(1,idx),samples(2,idx),samples(3,idx),samples(4,idx),samples(5,idx),samples(6,idx)));

if any(uval(1,:) > umax(1)) || any(uval(1,:) <  umin(1)) || ...
   any(uval(2,:) > umax(2)) || any(uval(2,:) <  umin(2) )|| ...
   any(uval(3,:) > umax(3)) || any(uval(3,:) <  umin(1)) 
        fprintf('Control Constraints are not met at sampling points!!!!!\n' )
else
    fprintf('Control Constraints are met at sampling points\n' )

end


% boresight and undesired direction may not be the same; Think about 
% n_I * T_IB * b_B where T_IB is the identity matrix. In this case our
% boresight would lie in the keep out zone which is not possible
b_B = [1;0;0];
n_I = [0;1;0];
coneAngle = 20*pi/180;
% plot unit sphere and keep-out-cone
plotConeOnUnitSphere(n_I, coneAngle,{'r'});
a = -1;
b = 1;
nSam = 1000;
sigma = a +(b-a)*rand(3,nSam);
Vfun = to_function(Wsol_re);
samples = [zeros(3,nSam);sigma];

idx = find(full(Vfun(samples(1,:),samples(2,:),samples(3,:),samples(4,:),samples(5,:),samples(6,:))) <= 0) ;

for k = 1:length(idx)
    T_IB = mrp2trafo_BI(sigma(:,idx(k)))';
    borSight_I = T_IB*b_B;
    plot3(borSight_I(1),borSight_I(2),borSight_I(3),'*k')
end


% evaluate the continous-time penalty sample points that lie in terminal set
penalties = full(penalty(samples(1,idx),samples(2,idx),samples(3,idx),samples(4,idx),samples(5,idx),samples(6,idx)));


% maximum value 
max_penalty = max(penalties);

if max_penalty > 0 
    fprintf('Something went wrong. The maximum penalty is %d\n',max_penalty )
else
    fprintf('Everything seems fine! The maximum penalty is %d\n',max_penalty )
end

% UNCOMMENT TO STORE THE INITIAL GUESS
% % store initial guess
% [coeffGuess,monoGuess] = poly2basis(remove_coeffs(sol.x,1e-6));
% 
% % store monomial basis
% [~,monoGuessV]  = poly2basis(remove_coeffs(sol.x(2),1e-6));
% [~,monoGuessS5] = poly2basis(remove_coeffs(sol.x(end),1e-6));
% 
% save('initGuess_termSet.mat','coeffGuess','monoGuess','monoGuessV','monoGuessS5')

% save ingredients for MPC
W_fun = to_function(Wsol_re);
V_fun = to_function(Vsol_re);
K_fun = to_function(Ksol_re);

% save for infinitesimal MPC
save('terminalIngredients.mat','W_fun','V_fun', 'K_fun', ... % polynomials
    'Q','R', 'gammaW',...                                             % weights
    'umin','umax','omegaMax1','omegaMax2','omegaMax3','J')   % parameter


