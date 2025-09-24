function plot_sphere_with_azimuth_and_elevation()
    % Create a figure with a 3D unit sphere and draw azimuth/elevation lines
    figure;
    hold on;
    
    % Plot a smooth unit sphere without mesh lines
    [sx, sy, sz] = sphere(1000);  % Higher resolution for smooth appearance
    surf(sx, sy, sz, 'FaceAlpha', 1, 'EdgeColor', 'none', 'FaceColor', [0.8 0.8 0.8]);
    axis equal;
 
    
    % Define parameters for the circle (equator) and azimuth/elevation lines
    theta = linspace(0, 2*pi, 100);  % Parameter for full circle
    
    % Equator (Azimuthal Circle)
    equator_x = cos(theta);
    equator_y = sin(theta);
    equator_z = zeros(size(theta));
    plot3(equator_x, equator_y, equator_z, 'k-');  % Thick black line for equator
    
    % Draw equally spaced azimuth lines (meridians)
    num_azimuth_lines = 2;                % Number of azimuth lines (meridians)
    azimuth_angles = linspace(0, 2*pi, num_azimuth_lines );  % Equally spaced angles from 0 to 2*pi
    azimuth_angles(end) = [];             % Remove the last angle as it's the same as the first (2*pi)

    for az = azimuth_angles
        x = cos(az) * sin(theta);
        y = sin(az) * sin(theta);
        z = cos(theta);
        plot3(x, y, z, 'k-', 'LineWidth', 0.5);  % Thin black lines for azimuth
    end
    
    % Draw elevation lines from equator to zenith (vertical meridian)
    elevation_angles = linspace(0, pi, 100);  % Vertical angle from equator to zenith
    x_elev = sin(elevation_angles);
    z_elev = cos(elevation_angles);
    y_elev = zeros(size(elevation_angles));
    plot3(x_elev, y_elev, z_elev, 'k-', 'LineWidth', 0.5);  % Thin black line for elevation
   
    
    % Set view and axis properties for better visualization
    view([120 30]);  % Adjust view angle for better perspective
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Azimuth and Elevation on the Unit Sphere');
    grid on;
    hold off;
end
