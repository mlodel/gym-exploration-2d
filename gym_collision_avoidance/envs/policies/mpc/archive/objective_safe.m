function [cost] =  objective_scenario(z_x, z_u, p, i, n_constraints)

%% Cost for ego-vehicle    
% states and inputs for ego vehicle
%            inputs               |               states
%                v   w  sv     x      y       theta      dummy

%% Inputs
a = z_u(1);
alpha = z_u(2);

network_a = p(1);
network_alpha = p(2);

%% States
x = z_x(1);
y = z_x(2);
theta = z_x(3);
v = z_x(4);
w = z_x(5);

cost = 0;

%% Safety filter objective
tail_weigth = 1e-2;
if(i == 1)
    cost = cost + (a - network_a)^2 + (alpha - network_alpha)^2;
else
    cost = cost + tail_weigth*a^2 + tail_weigth*alpha^2;
end

% %% Repulsive fields
% repulsive_weight = 1e-4;
% for l = 0 : n_constraints - 1
%     A1 = p(10+l*3+1); 
%     A2 = p(10+l*3+2);
%     b = p(10+l*3+3);
% 
%     g = A1*x+A2*y-b;
%     cost = cost + repulsive_weight * (1 / (g^2 + 0.001));
% end
    
end
