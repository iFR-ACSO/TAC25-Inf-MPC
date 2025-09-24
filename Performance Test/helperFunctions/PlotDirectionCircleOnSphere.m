function PlotDirectionCircleOnSphere(v)
    % Plots a circle with radius 1 along a specified direction `v` on the unit sphere.
    %
    % Input:
    %   v - 3D vector specifying the direction of the circle center (x, y, z)

    % Normalize the direction vector `v` to lie on the unit sphere
    v = v / norm(v);

    % Define the circle radius
    R = 1;                     % Radius of the circle to be drawn on the sphere
    c = v;                     % Center of the circle lies at the direction `v`

    % Generate points for a unit circle in the XY plane
    t = linspace(0, 2 * pi, 100);  % Parameter for circle points
    x = R * cos(t);                % X-coordinates of the circle
    y = R * sin(t);                % Y-coordinates of the circle
    z = zeros(size(t));            % Z-coordinates (initially zero for XY plane)

    % Stack the points as a 3xN matrix for transformation
    circle_points = [x; y; z];

    % Find a rotation axis and angle to align the XY plane circle with vector `v`
    w = cross([0; 0; 1], v);       % Cross product to find rotation axis
    if norm(w) > 1e-6
        w = w / norm(w);           % Normalize rotation axis if non-zero
    else
        w = [0; 1; 0];             % Default axis if `v` is along z-axis
    end
    b = acos(dot([0; 0; 1], v));   % Rotation angle between z-axis and `v`

    % Create rotation matrix using Rodrigues' formula
    R_matrix = expm(cpm(w) * b);

    % Rotate circle points to align with vector `v`
    rotated_circle = R_matrix * circle_points;

    % Translate the rotated circle to the intersection center `c`
    rotated_circle = rotated_circle + c;

    % Plot the unit sphere without mesh lines for visual context
    [sx, sy, sz] = sphere(100);        % Generate sphere with higher resolution
    surf(sx, sy, sz, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', [0.8 0.8 1]);
    hold on;

    % Plot the direction circle on the unit sphere
    plot3(rotated_circle(1, :), rotated_circle(2, :), rotated_circle(3, :), ...
          'k-', 'LineWidth', 0.5);     % Thin black line for the circle
    axis equal;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Circle on Unit Sphere in Specified Direction');
    hold off;
end

% Helper function to calculate the cross-product matrix
function C = cpm(w)
    % Computes the cross-product matrix for vector `w`
    C = [0, -w(3), w(2);
         w(3), 0, -w(1);
         -w(2), w(1), 0];
end
