function [cost] =  objective_scenario( x_R,u_R, p,i)
%% Cost for ego-vehicle    
% states and inputs for ego vehicle
%            inputs               |               states
%                v   w  sv     x      y       theta      dummy

v = u_R(1);
w = u_R(2);
%slack = u_R(3);

x = x_R(1);
y = x_R(2);
theta = x_R(3);

%% Online parameters

x_goal = p(1); y_goal = p(2); Wrepulsive = p(3); 

% Weightsvref
Wx = p(4);
Wy = p(5);
Walpha = p(6);
Wtheta = p(7);
Wa = p(8);
Ws = p(9);
% References
Wv = p(10);
Ww = p(11);

%% Total cost (0.9^i)*(
x_error = x - x_goal;
y_error = y - y_goal;

%% Parameters
    r_disc = p(27); disc_pos_0 = p(28);
    obst1_x = p(29); obst1_y = p(30); obst1_theta = p(31); obst1_major = p(32); obst1_minor= p(33);
    obst2_x = p(36); obst2_y = p(37); obst2_theta = p(38); obst2_major = p(39); obst2_minor= p(40);
    obst3_x = p(43); obst3_y = p(44); obst3_theta = p(45); obst3_major = p(46); obst3_minor= p(47);
    obst4_x = p(50); obst4_y = p(51); obst4_theta = p(52); obst4_major = p(53); obst4_minor= p(54);
    obst5_x = p(57); obst5_y = p(58); obst5_theta = p(59); obst5_major = p(60); obst5_minor= p(61);
    obst6_x = p(64); obst6_y = p(65); obst6_theta = p(66); obst6_major = p(67); obst6_minor= p(68);
    
    %% Collision Avoidance Constraints
    
    %% Obstacles
    % Obstacle 1
	deltaPos_disc_0_obstacle =  [sqrt((obst1_x-x)^2+(obst1_y-y)^2);
        sqrt((obst2_x-x)^2+(obst2_y-y)^2);
        sqrt((obst3_x-x)^2+(obst3_y-y)^2);
        sqrt((obst4_x-x)^2+(obst4_y-y)^2);
        sqrt((obst5_x-x)^2+(obst5_y-y)^2);
        sqrt((obst6_x-x)^2+(obst6_y-y)^2);];
    
    r_obstacles =[obst1_major;
        obst2_major;
        obst3_major;
        obst4_major;
        obst5_major;
        obst6_major;
        ];

    %% collision potential cost
    % define an empty cost vector
    obs_coll_cost = [];
    % cost for each obstacle
    nObs = 6;
    for jObs = 1 : nObs
        % if_else function
        d_c = deltaPos_disc_0_obstacle(jObs)/( r_disc + r_obstacles(jObs)) - 1;
        jObs_coll_cost = if_else(d_c>0.0, 0, -d_c);
        % add to the vector
        obs_coll_cost = [obs_coll_cost; jObs_coll_cost];
    end
    % the weight matrix
    Q_coll = Wrepulsive * eye(nObs);
    % the cost
    if nObs > 1
        cost_coll = obs_coll_cost' * Q_coll * obs_coll_cost;
    else
        cost_coll = 0;
    end


disToGoal = sqrt(x_error^2+y_error^2);
disToGoal   =   max(disToGoal, 0.2);      % in case arriving at goal posistion
max_v_range = 1.0;
max_w_range=6.0;

if length(u_R)>2
    if i == 20
        cost = Wx*x_error^2/disToGoal + Wy*y_error^2/disToGoal + Wv*v*v/max_v_range + Ww*w*w/max_w_range + Ws*u_R(3)*u_R(3) ;%+ Ws*slack*slack;%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
    else
        %distance_cost = if_else(disToGoal>1.0, 0, disToGoal);
        cost = Wv*v*v/max_v_range + Ww*w*w/max_w_range + Ws*u_R(3)*u_R(3) ;%+ Ws*slack*slack;%++1/(tcc_1+0.01)+1/(tcc_2+0.01)+1/(tcc_3+0.01)+1/(tcc_4+0.01)+1/(tcc_5+0.01)+1/(tcc_6+0.01);%+ Wv*(v-vref)*(v-vref);%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
    end
else
    if i == 20
        cost = Wx*x_error^2/disToGoal + Wy*y_error^2/disToGoal + Wv*v*v/max_v_range + Ww*w*w/max_w_range +cost_coll ;%+ Ws*slack*slack;%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
    else
        %distance_cost = if_else(disToGoal>1.0, 0, disToGoal);
        cost = Wv*v*v/max_v_range + Ww*w*w/max_w_range +cost_coll ;%+ Ws*slack*slack;%++1/(tcc_1+0.01)+1/(tcc_2+0.01)+1/(tcc_3+0.01)+1/(tcc_4+0.01)+1/(tcc_5+0.01)+1/(tcc_6+0.01);%+ Wv*(v-vref)*(v-vref);%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
    end
end

end
