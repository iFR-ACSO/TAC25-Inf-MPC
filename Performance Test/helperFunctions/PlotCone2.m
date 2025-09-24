%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%  Function taken from J.Breedens Repo
%  https://github.com/jbreeden-um/phd-code/tree/main/2022/AIAA%20Autonomous%20Attitude%20Reorientation/MATLAB
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [xp, yp, zp, translated] = PlotCone2(v, theta, color, d)
    % Plots a 3D cone aligned with vector `v`, originating from a point
    % along `v` at distance `d` from the origin, with an opening angle `theta`.
    % Ensures the cone lies within the unit sphere.
    % 
    % Inputs:
    %   v      - 3D vector for cone orientation
    %   theta  - cone opening angle in radians
    %   color  - color of the cone surface
    %   d      - distance along `v` from origin for cone origin point
    %
    % Outputs:
    %   xp, yp, zp - matrices of x, y, and z coordinates for plotting
    %   translated - matrix of 3D coordinates of the cone after transformation

    rho = 1;                  % Set the maximum radius of the cone base
    Alpha = 0.75;             % Transparency level for the cone surface
    v_unit = v / norm(v);     % Normalize `v` to unit vector
    p = d * v_unit;           % Set origin point along `v` at distance `d`
    
    ctheta = cos(theta);      % Compute cosine of the cone angle for calculations
    r1 = rho * sqrt(1 - ctheta^2); % Compute base radius based on cone angle

    % Adjust height `width` so that the cone base lies on the unit sphere
    width = sqrt(1 - (d^2 * ctheta^2)); % Adjusted cone height to keep base on unit sphere

    % Create radius distributions for smooth tapering
    rset1 = (0:1:20) * r1 / 20; 
    rset2 = sqrt((10:-1:0)) * r1 / sqrt(10);

    n = 50;                   % Set default number of points for plotting
    if nargout == 3           % If three outputs requested, use a finer grid
        n = 100;              % Increase the number of points for smoother plot
    end

    % Generate cone shape using cylinder for top and bottom sections
    [x1, y1, z1] = cylinder(rset1, n);
    [x2, y2, z2] = cylinder(rset2, n);
    
    z1 = z1 * width;          % Scale `z1` to match the adjusted height
    z2 = z2 * (rho - width) + width; % Scale `z2` for bottom section height
    
    % Combine the top and bottom sections
    x = [x1; x2];
    y = [y1; y2];
    z = [z1; z2];

    % Compute the rotation axis `w` and rotation angle `b`
    w = cross([0; 0; 1], v);  % `w` is the cross product of `z`-axis and `v`
    if norm(w) > 1e-6         % If `w` has significant magnitude
        w = w / norm(w);      % Normalize `w`
    else
        w = [0; 1; 0];        % Use default if `v` is aligned with z-axis
    end

    % Calculate the rotation angle `b`
    b = acos(dot([0; 0; 1], v_unit)); % Angle between `z`-axis and `v`

    % Construct rotation matrix `R`
    R = expm(cpm(w) * b);     % Rodrigues' rotation formula

    % Flatten x, y, z matrices for transformation
    s = size(x);              % Store original size for reshaping
    origin = [x(:)'; y(:)'; z(:)']; % Combine `x`, `y`, `z` into a matrix

    rotated = R * origin;     % Rotate points to align cone with `v`
    translated = rotated + p; % Translate points to the desired origin `p`

    % Reshape the rotated and translated coordinates back to original size
    xp = reshape(translated(1,:), s);
    yp = reshape(translated(2,:), s);
    zp = reshape(translated(3,:), s);

    % Plot the cone with specified color and transparency
    h = surf(xp, yp, zp, 'FaceAlpha', Alpha, 'EdgeAlpha', 0, 'FaceColor', color);
end

% Helper function to calculate the cross-product matrix
function C = cpm(w)
    % Computes the cross-product matrix for vector `w`
    C = [0, -w(3), w(2);
         w(3), 0, -w(1);
         -w(2), w(1), 0];
end
