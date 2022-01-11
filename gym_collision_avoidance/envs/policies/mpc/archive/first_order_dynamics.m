function x_next =  dynamic_scenario( x,u, p,dt )
%            inputs               |               states
%                acc   delta  sv     x      y       psi   v    s    dummy

    % ego-vehicle

    % integrator Runge-Ksutta integrator of order 4
    x_R_next = RK4(x, u, @continuous_dynamics_R, dt);
    x_next = x_R_next;
end

function xdot = continuous_dynamics_R ( x, u )
    v = u(1);
    w = u(2);
    theta =  x(3);

    xdot = [v * cos(theta);
            v * sin(theta);
            w;];
end