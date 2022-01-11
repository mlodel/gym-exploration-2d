theta = 0:0.01:4*pi;
theta = [theta 4*pi:-0.01:-4*pi];
theta = [theta -4*pi:-0.01:pi];

x = cos(theta);
y = sin(theta);

theta2 = atan2(y,x);

plot(theta,theta2);
hold on;

old_theta = theta2(1);
pi_increment_counter = 0;
orientation = [];
for i =1:1:length(theta2)

    if theta2(i) - old_theta > pi
        pi_increment_counter = pi_increment_counter - 1;
            
    elseif(theta2(i) - old_theta < -pi)
        pi_increment_counter = pi_increment_counter + 1;
    end
   orientation(i) = theta2(i) + 2*pi*pi_increment_counter;
   old_theta = theta2(i);
    
end

plot(theta,orientation);