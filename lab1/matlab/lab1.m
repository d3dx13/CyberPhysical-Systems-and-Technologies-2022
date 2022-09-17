% https://avtika.ru/kak-sostavit-matematicheskuyu-model-dvigatelya/

clear;
close all;

syms I w U R L C J Mc;

x = [I;w];
A = [
    -R/L, -C/L;
    C/J, 0
    ];
B = [
    1 / L, 0;
    0, 1 / J
    ];
C = [
    1, 0;
    0, 1
    ];
u_syms = [U; Mc];

x = [I;w];
A = [
    -R/L
    ];
B = [
    1 / L
    ];
C = [
    1
    ];
u_syms = [U];

% model = 
% Forward Euler (difference) discretization

syms z z_i s Ts;
s_new = (z - 1) / Ts;

Gc = C * (eye(size(A))*s - A)^(-1) * B

Gd = simplify(subs(Gc, s, s_new))
Gd = subs(Gd, z, z_i^(-1));
Gd = simplify(Gd)

% (L - L*z_i + R*Ts*z_i) * I = (Ts*z_i) * U
% (L + z_i * (R*Ts - L)) * I = (Ts * z_i) * U
% L * I = (Ts) * U*z_i - (R*Ts - L) * I*z_i
% I = (Ts / L) * U*z_i - ((R*Ts - L) / L) * I*z_i
% I(k) = (Ts / L) * U(k-1) - ((R*Ts - L) / L) * I(k-1)



