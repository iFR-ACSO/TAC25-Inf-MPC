function PlotConeSphereIntersection(v, theta, color)
    % Plots the intersection of a cone with the unit sphere.
    %
    % Inputs:
    %   v      - 3D vector for cone orientation
    %   theta  - cone opening angle in radians
    %   color  - color for the intersection circle plot

    % Normalize the cone axis vector `v`
    v = v / norm(v);

    % Calculate the intersection circle radius and center
    R = sin(theta);               % Radius of intersection circle
    c = cos(theta) * v;           % Center of intersection circle on the sphere

    % Generate points for a circle in the XY plane
    t = linspace(0, 2 * pi, 100); % Parameter for circle points
    x = R * cos(t);               % X-coordinates of the circle
    y = R * sin(t);               % Y-coordinates of the circle
    z = zeros(size(t));           % Z-coordinates (initially zero)

    % Stack the points as a 3xN matrix for transformation
    circle_points = [x; y; z];

    % Find rotation axis and angle to align the XY plane circle with vector `v`
    w = cross([0; 0; 1], v);      % Cross product to find rotation axis
    if norm(w) > 1e-6
        w = w / norm(w);          % Normalize rotation axis if non-zero
    else
        w = [0; 1; 0];            % Default axis if `v` is along z-axis
    end
    b = acos(dot([0; 0; 1], v));  % Rotation angle between z-axis and `v`

    % Create rotation matrix using Rodrigues' formula
    R_matrix = expm(cpm(w) * b);

    % Rotate circle points to align with vector `v`
    rotated_circle = R_matrix * circle_points;

    % Translate the rotated circle to the intersection center `c`
    rotated_circle = rotated_circle + c;

    % Plot the intersection circle on the unit sphere
    plot3(rotated_circle(1, :), rotated_circle(2, :), rotated_circle(3, :), ...
          'Color', color, 'LineWidth', 2);
    % hold on;

    % % Plot the unit sphere for visualization
    % [sx, sy, sz] = sphere(50);
    % surf(sx, sy, sz, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    % axis equal;
    % xlabel('X');
    % ylabel('Y');
    % zlabel('Z');
    % title('Intersection of Cone and Unit Sphere');
    % hold off;
end

% Helper function to calculate the cross-product matrix
function C = cpm(w)
    % Computes the cross-product matrix for vector `w`
    C = [0, -w(3), w(2);
         w(3), 0, -w(1);
         -w(2), w(1), 0];
end
