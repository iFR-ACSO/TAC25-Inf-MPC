%==========================================================================
%   Supplementary Material for "Safe-by-Design: Approximate Nonlinear Model 
%   Predictive Control with Realtime Feasibility" by 
%   Jan Olucak, Arthur Castello B. de Oliveira, and Torbjørn Cunis
%
%
%   Short Description:  Generate a CaSoS constraint for one keep-out zone.
%                       We first use the Matlab symbolic toolbox to setup 
%                       the constraint. We simplify the expression such 
%                       that we get rid of rational expressions to
%                       have a fully polynomial description. Afterward, we
%                       first check if we can convert the constraint to 
%                       CaSoS PS expression and then via sampling if the 
%                       constraint is correct.
%   
%   License: see License file in repository.   
%
%  Input: 
%           b_B       - instrument boresight direction in body frame
%           n_I       - keep out direction in inertial system
%           coneAnlge - open-angle of cone
%
% Output:   CaSoS constraint as g(sigma) \geq 0 i.e. we stay outside cone 
%           if inequality is fulfilled
%==========================================================================

function [keepOut_con_casos,g0_collected] = generate_KeepOut_constraint2(b_B,n_I,coneAngle)

% Define symbolic variables
syms sigma1 sigma2 sigma3 real
alpha = sym('alpha', 'real');

sigma = [sigma1 sigma2 sigma3]';

% instrument boresight points in body x-axis direction
b_B = b_B/norm(b_B);

% desired direction of keep-in in inertial coordinates
n_I = n_I/norm(n_I);

% Define the keep-in constraints in inertial coordinates
T_IB = mrp2trafo_BI(sigma)';

% g0 <= 0 (already re-written into SOS constraint i.e. contained if smaller
% than
g0 = n_I'*T_IB*b_B - cos(alpha) ;

% Get the numerator and denominator
[~,denominator] = numden(g0);

% Multiply g0 by the denominator to eliminate the rational fraction
g = simplify(g0 * denominator);

% Collect terms with respect to denominator
g0_collected = simplify(collect(g, denominator));

keepOut = matlabFunction(simplify(expand(g0_collected)));

% plot Cone
plotConeOnUnitSphere(n_I, coneAngle,{'r'});

%% check if the constraint can be converted to casos
x = casos.Indeterminates('x',6);

keepOut_con_casos = keepOut(coneAngle,x(4),x(5),x(6));

%% check correctness by smapling

newCon_Function = to_function(keepOut_con_casos);

% generate random attitude in MRP space
a = -1;
b =  1;

sigma = a +(b-a)*rand(3,1000);
sigma = sigma./vecnorm(sigma);

% evaluate constraint
newCon_Function_values  = full(newCon_Function(sigma(1,:),sigma(2,:),sigma(3,:)));

% check for all function values smaller or equal to zero i.e. points that 
% lie inside
idx = find(newCon_Function_values <= 0);

% plot safe attitudes in unit-sphere space i.e. we want that the borsight
% angle stays inside the cone
for k = 1:length(idx)
    % get transformation matrix
    T_IB = mrp2trafo_BI(sigma(:,idx(k)))';
    
    % transform to inertial frame
    borSight_I = T_IB*b_B;
    
    % plot
    plot3(borSight_I(1),borSight_I(2),borSight_I(3),'*k')

end

legend('unit-Sphere','x-axis','y-axis','z-axis','keep-out cone','samples (outside)')

end