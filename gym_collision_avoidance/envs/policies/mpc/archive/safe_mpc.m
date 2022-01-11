%% Clean up
clear; clc; close all; clearvars;
rng('shuffle');

%% Delete previous Solver
% Forces does not always code changes and might reuse the previous solution
try
FORCEScleanup('SafeMPCsolver','all');
catch
end

try
    rmdir('@FORCESproWS','s')
catch
end
try
    rmdir('SafeMPCsolver','s')
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
model.nh = n_constraints_per_region;           % number of inequality constraint functions

dt = 0.3;
model.npar = 10 + 3*n_constraints_per_region;         % number of parameters (10 to be sure)

%% Inequality constraints
% upper/lower variable bounds lb <= x <= ub
%            inputs               |               states
%                a      alpha     sv     x      y       theta      v w
%               
% model.lb = [ -2.0,  -1.0,   0, -200,   -200,    -1.5*pi,        0  ];
% model.ub = [ +2.0,  +1.0,   800, +200,   +200,    +1.5*pi,    inf];

% Lower limits for robot
lb_R = [ -2,  -2, 0, -50,   -50,    -1.5*pi, -0.01, -2.0];
model.lb = lb_R;

% Upper limits for robot
ub_R = [ +2,  +2, +inf, +50,   +50,    +1.5*pi, 1.1, 2.0];

model.ub =ub_R;
%%
for i=1:model.N
    
    model.objective{i} = @(z, p) objective_safe(z(4:8), z(1:3), p, i, n_constraints_per_region);
    %% Objective function
%     if(i == 1)
%         model.objective{i} = @(z, p) (z(1:2) - p(1:2))'*(z(1:2) - p(1:2)); % p(1) - p(2) is network input!
%     else
%         model.objective{i} = @(z, p) (z(1:2)'*z(1:2)); % Now it is optimizing...
%     end
    
    model.ineq{i} = @(z,p) inequality_safe(z(4: 8),z(1: 3), p, n_constraints_per_region);

    %% Upper/lower bounds For road boundaries
    model.hu{i} = [0*ones(1, n_constraints_per_region)];   
    model.hl{i} = [-Inf*ones(1, n_constraints_per_region)];
end
%% Dynamics, i.e. equality constraints 
%model.objective = @(z, p) objective_scenario_try(z, p);
model.eq = @(z, p) dynamic_scenario(z(4:8),z(1:3), p, dt);

model.E = [zeros(5,3), eye(5)];

%% Initial and final conditions
% Initial condition on vehicle states

model.xinitidx = 4:8; % use this to specify on which variables initial conditions are imposed
%model.xfinal = 0; % v final=0 (standstill), heading angle final=0?
%model.xfinalidx = 6; % use this to specify on which variables final conditions are imposed

%% Define solver options
codeoptions = getOptions('SafeMPCsolver');
codeoptions.maxit = 500;   % Maximum number of iterations
codeoptions.printlevel = 1 ; % Use printlevel = 2 to print progress (but not for timings)
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
