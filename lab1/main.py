import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

data_read = np.genfromtxt('dataset/testLab1Var7.csv', delimiter=',')

time = data_read[:, 0]
time = time[:, np.newaxis]
current = data_read[:, 1]
current = current[:, np.newaxis]
voltage = data_read[:, 2]
voltage = voltage[:, np.newaxis]

fig, (ay1, ay2) = plt.subplots(2, 1, sharex=True)
T_per = 0.1
ay1.plot(time[time < 2 * T_per], voltage[time < 2 * T_per])
ay1.grid()
ay1.set_xlabel('time, s')
ay1.set_ylabel('voltage, V')
ay2.plot(time[time < 2 * T_per], current[time < 2 * T_per])
ay2.grid()
ay2.set_xlabel('time, s')
ay2.set_ylabel('current, A')
plt.show()
fig.savefig('Recieved data(part)')

X = np.concatenate([voltage[0:len(voltage) - 2], current[0:len(current) - 2]], axis=1)
Y = current[1:len(current) - 1]
K = np.dot(np.dot(LA.inv(np.dot(X.T, X)), X.T), Y)

print(K)

Td = 0.001
R = 1 / K[0] * (1 - K[1])
T = -Td / np.log(K[1])
L = T * R

print(L, R)

current_est = X.dot(K)

fig, ax = plt.subplots(1, 1)
plt.plot(time[time < T_per], current[time < T_per])
plt.plot(time[time < T_per], current_est[time[0:len(current) - 2] < T_per])
ax.grid()
ax.set_xlabel('time, s')
ax.set_ylabel('current, A')
plt.show()
fig.savefig('Compared data(part)')

R_est = []
L_est = []

n = 1000
for i in range(0, n - 1, 1):
    ind = (time >= T_per * i) & (time <= T_per * (i + 1))
    new_current = current[ind]
    new_current = new_current[:, np.newaxis]
    new_voltage = voltage[ind]
    new_voltage = new_voltage[:, np.newaxis]

    X = np.concatenate([new_voltage[1:len(new_voltage) - 1], new_current[0:len(new_current) - 2]], axis=1)
    Y = current[1:len(new_current) - 1]
    K = np.dot(np.dot(LA.inv(np.dot(X.T, X)), X.T), Y)

    if K[1] > 0:
        R = 1 / K[0] * (1 - K[1])
        T = -Td / np.log(K[1])
        R_est.append(R)
        L_est.append(T * R)

R_est = np.array(R_est)
L_est = np.array(L_est)

print('Mean value of R: ', np.mean(R_est), ' Ohm')
print('Standart deviation of R: ', np.std(R_est))
print('Mean value of L = ', np.mean(L_est), ' Hn')
print('Standart deviation of R: ', np.std(L_est))
