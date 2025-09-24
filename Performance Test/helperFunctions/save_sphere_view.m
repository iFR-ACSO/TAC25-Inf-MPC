function save_sphere_view(az, el, filename, x_sol_all, iter_all, simStepSize)
    % --- constants (identical for both views) ---
    coneAngle = 20*pi/180;
    b_B = [1;0;0];
    n_I = [0;1;0];

    % figure/axes geometry (identical for both exports)
    fig = figure('Color','w','Units','pixels','Position',[100 100 900 900]); % square fig
    ax  = axes('Parent',fig,'Units','pixels','Position',[80 80 740 740]);    % square axes

    % projection & aspect (identical!)
    axis(ax, [-1.1 1.1 -1.1 1.1 -1.1 1.1]);  % same limits
    axis(ax, 'vis3d');                        % lock data aspect across rotations
    axis(ax, 'equal');                        % sphere stays round
    axis(ax, 'off');                          % no ticks/box
    set(ax,'Projection','orthographic');      % no perspective scaling
    daspect(ax,[1 1 1]);  pbaspect(ax,[1 1 1]);

    % camera (identical except azimuth/elevation)
    camtarget(ax,[0 0 0]);                    % look at the origin
    camva(ax,8);                              % same view angle
    camup(ax,[0 0 1]);                        % stable up direction

    % ---- draw your scene ----
    plotConeOnUnitSphere_mesh(n_I, coneAngle, {'r'}); hold(ax,'on');

    % trajectories (your loop, unchanged except drawing into ax)
    skipdata = 1600;
    for jj = 1:length(x_sol_all)
        x_sol_vec = x_sol_all{jj};
        x_sol_vec = x_sol_vec(:,1:iter_all{jj});
        % endpoints
        b_I0 = mrp2trafo_BI(x_sol_vec(4:6,1))'*b_B;
        b_ITend = mrp2trafo_BI(x_sol_vec(4:6,end))'*b_B;
        plot3(ax,b_I0(1),b_I0(2),b_I0(3),'*','Color','g','LineWidth',2);
        plot3(ax,b_ITend(1),b_ITend(2),b_ITend(3),'b*');

        % trajectory subsampling
        idx = 1:skipdata:length(x_sol_vec);
        b_IT = zeros(3,numel(idx));
        for k = 1:numel(idx)
            b_IT(:,k) = mrp2trafo_BI(x_sol_vec(4:6,idx(k)))'*b_B;
        end
        plot3(ax,b_IT(1,:),b_IT(2,:),b_IT(3,:),'Color','g');
    end
    axis off
        view(ax, az, el);                         % <-- only thing that changes
    % --- export (identical for both) ---
    % Vector PDF is best; if you truly want raster inside PDF, set 'image'.
    exportgraphics(ax, filename, ...
        'ContentType','image', ...     
       'BackgroundColor', 'none', ...
       'Resolution', 600);               % Adjust resolution (300–600 is typica     % keeps transparency for LaTeX
    close(fig);
end
