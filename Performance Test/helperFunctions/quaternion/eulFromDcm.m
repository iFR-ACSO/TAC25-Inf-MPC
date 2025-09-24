function [phi,theta,psi] = eulFromDcm(T)

    % extract entries
    T11 = T(1,1);
    T33 = T(3,3);
    T12 = T(1,2);
    T13 = T(1,3);
    T23 = T(2,3);

    phi     = atan2(T23,T33);
    theta   = -asin(T13);
    psi     = atan2(T12,T11);

end