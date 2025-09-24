clc
clear


% from euler to dcm
phi = 42*pi/180;
theta = 91*pi/180;
psi = 10*pi/180;


T_BA = dcm321(phi,theta,psi);

[phi0,theta0,psi0] = eulFromDcm(T_BA);

q_BA = mat2quat(T_BA);

T_BA2 = quat2mat(q_BA);


T_BA-T_BA2

% from quaternion
q0 = [0.4 0.3 0.1 0.6];

q0 = q0/norm(q0);

T_BA = quat2mat(q0);

[phi1,theta1,psi1] = eulFromDcm(T_BA);

T_BA0 =  dcm321(phi1,theta1,psi1);

T_BA0-T_BA

