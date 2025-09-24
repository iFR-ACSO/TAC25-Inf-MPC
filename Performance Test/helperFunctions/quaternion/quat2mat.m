% function computes the transformation matrix from a given attitude
% quaternion
% keep in min the quaternion q_AB describes the attitude of frame A w.r.t
% to frame B; similar T_AB describes attitude from A to B but is the
% transformation matrix from B to A!

function [T_AB] = quat2mat(quat_AB)

qvec = quat_AB(1:3);
q4   = quat_AB(end);


T_AB = (2*q4^2 -1)*eye(3)+2*(qvec*qvec'-q4*cpm(qvec));

end