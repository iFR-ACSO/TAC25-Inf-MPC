function q = mat2quat(T)
    % Validate input
    if size(T,1) ~= 3 || size(T,2) ~= 3
        error('Input T must be a 3x3 matrix');
    end

    % Initialize quaternion vector
    q = zeros(4, 1);

    % Calculate traces
    T11 = T(1,1);
    T22 = T(2,2);
    T33 = T(3,3);
    T12 = T(1,2);
    T21 = T(2,1);
    T13 = T(1,3);
    T31 = T(3,1);
    T23 = T(2,3);
    T32 = T(3,2);

    % Case 1
    if T11 > T22 && T11 > T33
        q(1) = 0.5 * sqrt(1 + T11 - T22 - T33);
        q(2) = (1 / (4 * q(1))) * (T12 + T21);
        q(3) = (1 / (4 * q(1))) * (T13 + T31);
        q(4) = (1 / (4 * q(1))) * (T23 - T32);

    % Case 2
    elseif T22 > T11 && T22 > T33
        q(2) = 0.5 * sqrt(1 + T22 - T11 - T33);
        q(1) = (1 / (4 * q(2))) * (T12 + T21);
        q(3) = (1 / (4 * q(2))) * (T23 + T32);
        q(4) = (1 / (4 * q(2))) * (T31 - T13);

    % Case 3
    elseif T33 > T11 && T33 > T22
        q(3) = 0.5 * sqrt(1 + T33 - T11 - T22);
        q(1) = (1 / (4 * q(3))) * (T13 + T31);
        q(2) = (1 / (4 * q(3))) * (T23 + T32);
        q(4) = (1 / (4 * q(3))) * (T12 - T21);

    % Case 4
    else
        q(4) = 0.5 * sqrt(1 + T11 + T22 + T33);
        q(1) = (1 / (4 * q(4))) * (T23 - T32);
        q(2) = (1 / (4 * q(4))) * (T31 - T13);
        q(3) = (1 / (4 * q(4))) * (T12 - T21);
    end
end
