function x_next =  dynamic_scenario_drone( x,u, p,dt )

    % integrator Runge-Kutta integrator of order 4
    x_R_next = RK4(x, u, @continuous_dynamics_R, dt);
    x_next = x_R_next;
end

function xdot = continuous_dynamics_R ( x, u )
    a_x = u(1);
    a_y = u(2);
    
    theta =  x(3);
    v_x = x(4);
    v_y = x(5);
    xdot = [v_x * cos(theta) - v_y * sin(theta);
            v_x * sin(theta) + v_y * cos(theta);
            0;
            a_x;
            a_y];
end