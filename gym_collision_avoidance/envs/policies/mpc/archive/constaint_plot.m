close all;
clc;
x = -10:0.1:10;

%plot(x,1/0.98*(0.22*x+2.3))
hold on;
p =[4.4,8.75];
plot(p(1),p(2),'m*')
plot(4.4,8.75,'r+')
plot(-0.2,9.79,'r+')
plot(-1.32,5.1,'r+')

% current state
current_state = [1.54,6.92];
plot(1.54,6.92,'bo')

% orientation
theta = 0.1;
[delta_x, delta_y] = pol2cart(theta,1);
plot([current_state(1), current_state(1) + delta_x], [current_state(2), current_state(2) + delta_y],'go' )
hold on;
ylim([-12,12])

r_max = 2.4;
obstacles1 = current_state' + [cos(theta) -sin(theta);sin(theta),cos(theta)]*[r_max;0];
obstacles2 = current_state' + [cos(theta) -sin(theta);sin(theta),cos(theta)]*[-r_max;0];
obstacles3 = current_state' + [cos(theta) -sin(theta);sin(theta),cos(theta)]*[0;r_max];
obstacles4 = current_state' + [cos(theta) -sin(theta);sin(theta),cos(theta)]*[0;-r_max];

plot(obstacles1(1),obstacles1(2),'b+')
plot(obstacles2(1),obstacles2(2),'b+')
plot(obstacles3(1),obstacles3(2),'b+')
plot(obstacles4(1),obstacles4(2),'b+')

p1 = obstacles1 - current_state';
p2 = obstacles2 - current_state';
p3 = obstacles3 - current_state';
p4 = obstacles4 - current_state';

A1 = -1*p1/norm(p1);
A2 = -1*p2/norm(p2);
A3 = -1*p3/norm(p3);
A4 = -1*p4/norm(p4);

b1 = A1'*obstacles1;
b2 = A2'*obstacles2;
b3 = A3'*obstacles3;
b4 = A4'*obstacles4;

plot(x,(-x*A1(1)+b1)/A1(2),'--')
plot(x,(-x*A2(1)+b2)/A2(2),'--')
plot(x,(-x*A3(1)+b3)/A3(2),'--')
plot(x,(-x*A4(1)+b4)/A4(2),'--')

b1 = A1'*obstacles1+0.5;
b2 = A2'*obstacles2+0.5;
b3 = A3'*obstacles3+0.5;
b4 = A4'*obstacles4+0.5;

plot(x,(-x*A1(1)+b1)/A1(2),'g--')
plot(x,(-x*A2(1)+b2)/A2(2),'g--')
plot(x,(-x*A3(1)+b3)/A3(2),'g--')
plot(x,(-x*A4(1)+b4)/A4(2),'g--')

constraint1 = current_state(1)*A1(1)-b1+A1(2)*current_state(2)
constraint2 = current_state(1)*A2(1)-b2+A2(2)*current_state(2)
constraint3 = current_state(1)*A3(1)-b3+A3(2)*current_state(2)
constraint4 = current_state(1)*A4(1)-b4+A4(2)*current_state(2)

obstacle_constraint1 = p(1)*A1(1)-b1+A1(2)*p(2)
obstacle_constraint2 = p(1)*A2(1)-b2+A2(2)*p(2)
obstacle_constraint3 = p(1)*A3(1)-b3+A3(2)*p(2)
obstacle_constraint4 = p(1)*A4(1)-b4+A4(2)*p(2)