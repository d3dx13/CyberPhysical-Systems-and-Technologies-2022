% https://www.gaussianwaves.com/2014/06/linear-models-least-squares-estimator-lse/amp/

clear;
close all;

syms k n positive;

syms y; % [n,1]
syms X; % [n,k]
syms O; % [k,1]
% syms e; % [k,1] was [n,1]

% y = X * (O + e)
% y = X * O + X * e
% X * e = y -  X * O 
% X_si * X * e = X_si * y -  X_si * X * O 
% e ~= X_si * y -  O 

syms X_si; % [k,n]

e = X_si * y -  O; % [k,n]*[n,1] - [k,1] = [k,1]

% e'e = (X_si * y -  O)' * (X_si * y -  O)
% e'e = ((X_si * y)' -  O') * (X_si * y -  O)
% e'e = (y' * X_si' -  O') * (X_si * y -  O)
% e'e = (y' * X_si' * X_si * y) - (y' * X_si' * O) - (O' * X_si * y) + (O'O)
% e'e = (y' * X_si' * X_si * y) - (y' * X_si' * O) - (O' * X_si * y) + (O'O)
% S = e'e
% dS/dO = - (y' * X_si')' - (X_si * y) + 2*O
% dS/dO = - (X_si * y) - (X_si * y) + 2*O
% dS/dO = - 2*(X_si * y) + 2*O
% 0 = dS/dO = - 2*(X_si * y) + 2*O = 0
% 2*O = 2*(X_si * y)
% O = (X_si * y)

% e = (X_si * y -  O)
% X [n, k] * e [k, 1] = y -  X * O : [n, 1]

% y(i) = K1 * y(i-1) + K2 * x(i-1)
% y(i-1) = K1 * y(i-2) + K2 * x(i-2)

