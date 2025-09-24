function [xparr,yparr,zparr,cone] = plotConeOnUnitSphere(n_I, coneAngle,color)



% figure('Name','Keep-In/Out Cone(s) on unit sphere')
% plot unit sphere
[x,y,z] = sphere(40);
% Define the start (white) and end (very light gray) colors
start_color = [1, 1, 1];        % White
end_color = [0.9, 0.9, 0.9];    % Very light gray

% Create a colormap with 256 entries that transition from white to light gray
colormap_size = 256;
gray_colormap = [linspace(start_color(1), end_color(1), colormap_size)', ...
                 linspace(start_color(2), end_color(2), colormap_size)', ...
                 linspace(start_color(3), end_color(3), colormap_size)'];

% Apply the colormap
colormap(gray_colormap);



set(gca, 'YDir','reverse')
set(gca, 'ZDir','reverse')
hold on
% % plot coordinate system
% plot3([0 0 1], [0 0 0], [0 0 0], 'r','LineWidth',2)
% plot3([0 0 0], [0 0 1], [0 0 0], 'g','LineWidth',2)
% plot3([0 0 0], [0 0 0], [0 0 1], 'b','LineWidth',2)
% % plot cone

xparr = [];
yparr = [];
zparr = [];
for k = 1:length(coneAngle)
   % [xp,yp,zp,cone] = PlotCone(n_I(:,k),coneAngle(k),color{k});
    
   % xparr = [xparr; xp];
   %  yparr = [yparr; yp];
   %   zparr = [zparr; zp];
    % PlotConeSphereIntersection(n_I(:,k),coneAngle(k), color{k})
end
hold on
f = surf(x,y,z,'FaceColor',[1 1 1],'EdgeColor',[0.6 0.6 0.6],'FaceAlpha',0.5,'EdgeAlpha',1);
axis equal
% shading faceted
[xp,yp,zp,cone] = PlotCone_hat(n_I(:,k),coneAngle(k),color{k});

xparr = reshape(xparr,[length(coneAngle) size(xparr,1)/length(coneAngle) size(xparr,2)]);
yparr = reshape(yparr,[length(coneAngle) size(yparr,1)/length(coneAngle) size(yparr,2)]);
zparr = reshape(zparr,[length(coneAngle) size(zparr,1)/length(coneAngle) size(zparr,2)]);

grid off

xlabel('x_I')
ylabel('y_I')
zlabel('z_I')
box on
