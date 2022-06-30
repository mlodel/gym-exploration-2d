function [cost] =  objective_scenario( x_R,u_R, p,i,N)
%% Cost for ego-vehicle    
% states and inputs for ego vehicle
%            inputs               |               states
%                v   w  sv     x      y       theta      dummy

a = u_R(1);
alpha = u_R(2);

x = x_R(1);
y = x_R(2);
theta = x_R(3);
v = x_R(4);
w = x_R(5);
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

c1 = p(59);
c2 = p(60);
c3 = p(61);
c4 = p(62);
c5 = p(63);
c6 = p(64);
d = p(65);
w_cost = p(66);

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

    obs_lambda = 10.0;
    
    %field1 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_1 - (obst1_major+r_disc+0.05))),10^3));
    %field2 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_2 - (obst2_major+r_disc+0.05))),10^3));
    %field3 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_3 - (obst3_major+r_disc+0.05))),10^3));
    %field4 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_4 - (obst4_major+r_disc+0.05))),10^3));
    %%field5 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_5 - (obst5_major+r_disc+0.05))),10^3));
    %field6 =  1.0 / (1.0+min(exp(obs_lambda*(deltaPos_disc_0_obstacle_6 - (obst6_major+r_disc+0.05))),10^3));
    
%     delta_vx_1 = v*cos(theta)-obst1_vx;
%     delta_vy_1 = v*sin(theta)-obst1_vy;
%     a = delta_vx_1^2+delta_vy_1^2;
%     b = (obst1_x-x)*(delta_vx_1)+(obst1_y-y)*(delta_vy_1)
%     c = (obst1_x-x)^2+(obst1_y-y)^2 - (r_disc + r_obstacles(1))^2
% 
%     tcc_1 = if_else(b^2-a*c<0.0, 0, b^2-a*c);
    

%cost = Wx*x_error*x_error + Wy*y_error*y_error + Wv*v*v +Ww*w*w + Ws*sv*sv; % Wv*v*v +Ww*w*w  + Ws*sv*sv
%if i == 20
%    cost = Wx*x_error*x_error + Wy*y_error*y_error + w_cost*(c1 + c2*x + c3*y+ c4*x*x + c5*x*y + c6*y*y + d)+ Wv*v*v +Ww*w*w + Ws*sv*sv+Wrepulsive*(field1 + field2 +field3 + field4+field5 + field6);
%else
%    cost = Wx*x_error*x_error + Wy*y_error*y_error + Wv*v*v +Ww*w*w + Ws*sv*sv+Wrepulsive*(field1 + field2 +field3 + field4+field5 + field6);
%end

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


%% New version

max_v_range = 2.0;
max_w_range=4.0;
max_acc_range = 2.0;
max_alpha_range=2.0;

if length(u_R)>2
    if i == N
        cost = Wx*x_error^2/disToGoal + Wy*y_error^2/disToGoal + Wv*v*v/max_v_range + Wv*w*w/max_v_range + Ws*u_R(3)*u_R(3);%+ Wv*v*v + Ww*w*w+Wrepulsive*(field1+field2+field3+field4+field5+field6);
    else
        cost = Wa*a*a/max_acc_range +Walpha*alpha*alpha/max_alpha_range   + Wv*v*v/max_v_range + Wv*w*w/max_v_range+ Ws*u_R(3)*u_R(3);%+1/(tcc_1+0.01)+1/(tcc_2+0.01)+1/(tcc_3+0.01)+1/(tcc_4+0.01)+1/(tcc_5+0.01)+1/(tcc_6+0.01);%+ Wv*(v-vref)*(v-vref);%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
    end
else
    if i == N
        cost = Wx*x_error^2/disToGoal + Wy*y_error^2/disToGoal + Wv*v*v/max_v_range + Wv*w*w/max_v_range +cost_coll;%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
    else
        cost = Wa*a*a/max_acc_range +Walpha*alpha*alpha/max_alpha_range + Wv*v*v/max_v_range + Wv*w*w/max_v_range +cost_coll;%++1/(tcc_1+0.01)+1/(tcc_2+0.01)+1/(tcc_3+0.01)+1/(tcc_4+0.01)+1/(tcc_5+0.01)+1/(tcc_6+0.01);%+ Wv*(v-vref)*(v-vref);%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
    end
end
%% OLD version

%max_v_range = 10.0;
%max_w_range=12.0;

%if i == 20
%    cost = 8.0*(Wx*x_error^2/disToGoal + Wy*y_error^2/disToGoal)+ w_cost*(c1 + c2*x + c3*y+ c4*x*x + c5*x*y + c6*y*y + d) + Ws*sv*sv;%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
%else
%    cost = Wv*a*a/max_v_range +Ww*alpha*alpha/max_w_range + Ws*sv*sv ;%+ Wv*(v-vref)*(v-vref);%+Wrepulsive*(field1+field2+field3+field4+field5+field6);
%end

end
