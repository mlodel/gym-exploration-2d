%% Clean up
clear; clc; close all; clearvars;
rng('shuffle');

%% Delete previous Solver
% Forces does not always code changes and might reuse the previous solution
try
FORCEScleanup('StaticMPCsolver','all');
catch
end

try
    rmdir('@FORCESproWS','s')
catch
end
try
    rmdir('StaticMPCsolver','s')
catch
end
% 
%% Some utility functions
deg2rad = @(deg) deg/180*pi; % convert degrees into radians
rad2deg = @(rad) rad/pi*180; % convert radians into degrees

%% Problem dimensions
n_constraints_per_region = 4;
model.N = 15;            % horizon length
model.nvar = 8;          % number of variables
model.neq= 5;            % number of equality constraints
model.nh = 6 + n_constraints_per_region;            % number of inequality constraint functions
n_other_param = 71;
dt = 0.1;
model.npar =  n_other_param +12;          % number of parameters

%% Inequality constraints
% upper/lower variable bounds lb <= x <= ub
%            inputs               |               states
%                a      alpha     sv     x      y       theta      v w
%               
% model.lb = [ -2.0,  -1.0,   0, -200,   -200,    -1.5*pi,        0  ];
% model.ub = [ +2.0,  +1.0,   800, +200,   +200,    +1.5*pi,    inf];

% Lower limits for robot
lb_R = [ -2  -3, 0, -50,   -50,    -1.5*pi, 0, -3.0];
%lb_R = [ 0, -2, 0, -50,   -50,    -1.5*pi];
model.lb = lb_R;

% Upper limits for robot
ub_R = [ +2,  +3, 10000, +50,   +50,    +1.5*pi, 2.0, 3.0];
%ub_R = [ 2,  2, 10000, +50,   +50,    +1.5*pi];

model.ub =ub_R;
%%
for i=1:model.N
    %% Objective function
    model.objective{i} = @(z, p) objective_scenario(z(4:8),z(1: 3), p,i,model.N); 

    model.ineq{i} = @(z,p) inequality_constr(z(4: 8),z(1: 3), p, i);

    %% Upper/lower bounds For road boundaries
   model.hu{i} = [+Inf, +Inf, +Inf, +Inf, +Inf, +Inf, 0*ones(1, n_constraints_per_region)]; % 2*3 for ellips 1 for binary
   model.hl{i} = [1, 1, 1, 1, 1, 1, -Inf*ones(1, n_constraints_per_region)];
end
%% Dynamics, i.e. equality constraints 
%model.objective = @(z, p) objective_scenario_try(z, p);
model.eq = @(z, p) dynamic_scenario(z(4: 8),z(1: 3), p, dt);
%model.eq = @(z, p) first_order_dynamics(z, p);

model.E = [zeros(5,3), eye(5)];
%model.E = [zeros(3,3), eye(3)];

%% Initial and final conditions
% Initial condition on vehicle states

model.xinitidx = 4:8; % use this to specify on which variables initial conditions are imposed
%model.xfinal = 0; % v final=0 (standstill), heading angle final=0?
%model.xfinalidx = 6; % use this to specify on which variables final conditions are imposed

%% Define solver options
codeoptions = getOptions('StaticMPCsolver');
codeoptions.maxit = 500;   % Maximum number of iterations
codeoptions.printlevel = 0 ; % Use printlevel = 2 to print progress (but not for timings)
codeoptions.optlevel = 3;   % 0: no optimization, 1: optimize for size, 2: optimize for speed, 3: optimize for size & speed
codeoptions.timing = 1;
codeoptions.overwrite = 1;
codeoptions.mu0 = 20;
codeoptions.cleanup = 0;
codeoptions.BuildSimulinkBlock = 0;
%codeoptions.noVariableElimination = 1;
%codeoptions.nlp.TolStat = 1E-3;     % infinity norm tolerance on stationarity
%codeoptions.nlp.TolEq   = 1E-3;     % infinity norm of residual for equalities
%codeoptions.nlp.TolIneq = 1E-3;     % infinity norm of residual for inequalities
%codeoptions.nlp.TolComp = 1E-3;     % tolerance on complementarity conditions

FORCES_NLP(model, codeoptions);
