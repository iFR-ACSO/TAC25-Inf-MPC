function q_BA = initQuatFromEul(phi,theta,psi)

% generate DCM 
T_BA = @(phi, theta, psi) [
                            cos(theta) * cos(psi),                                      cos(theta) * sin(psi),                                    -sin(theta);
                            sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),     sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),   sin(phi) * cos(theta);
                            cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),     cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),   cos(phi) * cos(theta)
                        ];

T_BA = T_BA(phi,theta,psi)';

% DCM to quat
q_BA = mat2quat(T_BA);


end