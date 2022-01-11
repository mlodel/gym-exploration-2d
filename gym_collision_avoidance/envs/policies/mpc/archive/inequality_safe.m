function ineq_constr =  inequality_safe(z_x, z_u, p, n_constraints)

% states and inputs for ego vehicle

    x = z_x(1);
    y = z_x(2);
    theta = z_x(3);
    
    %% Parameters
    %r_disc = p(27); disc_pos_0 = p(28);

    %% Collision Avoidance Constraints
% 	R_car= [cos(theta), -sin(theta); sin(theta), cos(theta)];
% 	CoG = [x;y];
% 
% 	shift_0 = [disc_pos_0; 0];
% 
%     % Car disc positions
% 	position_disc_0 = CoG+R_car*shift_0;
    
    %% Static Constraints
    ineq_constr = [];
    for l = 0 : n_constraints - 1
        A1 = p(10+l*3+1); 
        A2 = p(10+l*3+2);
        b = p(10+l*3+3);
        
        ineq_constr = [ineq_constr;A1*x+A2*y-b];
    end
end
    
