% Closed-Loop Differential-Drive Mobile Robot Kinematics & PID Control
clear; clc; close all;

% Robot physical parameters
r = 0.05;       % Wheel radius (m)
L = 0.24;       % Wheelbase distance (m)
dt = 0.01;      % Simulation time step (s)
T = 15.0;       % Total simulation duration (s)
t = 0:dt:T;

% Target reference waypoint
x_ref = 2.5;
y_ref = 1.8;

% PID controller gains
Kp_v = 1.2; Ki_v = 0.05; Kd_v = 0.1;
Kp_w = 3.5; Ki_w = 0.02; Kd_w = 0.2;

% Initial state [x; y; theta]
state = [0; 0; 0];
history = zeros(3, length(t));

err_v_int = 0; err_v_prev = 0;
err_w_int = 0; err_w_prev = 0;

for k = 1:length(t)
    x = state(1); y = state(2); theta = state(3);
    history(:, k) = state;

    dx = x_ref - x;
    dy = y_ref - y;
    dist_err = hypot(dx, dy);
    angle_ref = atan2(dy, dx);
    angle_err = atan2(sin(angle_ref - theta), cos(angle_ref - theta));

    % PID linear velocity control
    err_v_int = err_v_int + dist_err * dt;
    err_v_der = (dist_err - err_v_prev) / dt;
    v = Kp_v * dist_err + Ki_v * err_v_int + Kd_v * err_v_der;
    v = min(max(v, -0.8), 0.8);
    err_v_prev = dist_err;

    % PID angular velocity control
    err_w_int = err_w_int + angle_err * dt;
    err_w_der = (angle_err - err_w_prev) / dt;
    w = Kp_w * angle_err + Ki_w * err_w_int + Kd_w * err_w_der;
    w = min(max(w, -2.5), 2.5);
    err_w_prev = angle_err;

    % Kinematic state update
    state(1) = x + v * cos(theta) * dt;
    state(2) = y + v * sin(theta) * dt;
    state(3) = theta + w * dt;
end

fprintf('Final Position: (%.3f, %.3f) | Steady-State Error: %.4f m\n', ...
    state(1), state(2), hypot(x_ref - state(1), y_ref - state(2)));
