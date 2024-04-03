import numpy as np
import matplotlib.pyplot as plt

maxTime = 10
L = 0.8
g = 10
dt = 1/240 # pybullet simulation step
q0 = 0.5   # starting position (radian)
logTime = np.arange(0.0, maxTime, dt)


def rp(x):
    y = np.array([x[1], -g / L * np.sin(x[0])])
    return y[-1]


def symplectic_euler(rp, x0, dt, t):
    N = len(t)
    x = np.zeros((N, len(x0)))
    x[0] = x0
    v = np.zeros((N, len(x0)))
    v[0] = 0

    for i in range(1, N):
        v[i] = v[i - 1] + rp(x[i - 1]) * dt
        x[i] = x[i - 1] + v[i] * dt
    return x


theta = symplectic_euler(rp, [q0, 0], dt, logTime)
logTheta = theta[:, 0]

plt.grid(True)
plt.plot(logTime, np.sin(logTheta), label="theorPos")
plt.legend()
plt.show()

# 1 избавиться от затухания синуса
# 2 понять, откуда берется невязка между траекторией симулятора и моделью
# и попытаться ее минимизировать + посчитать нормы L2' и Linf
# есть как минимум два источника невязки
# идеал = l2(1.8766961224229702e-06), linf(4.531045163083669e-06)