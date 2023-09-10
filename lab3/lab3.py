import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


def Rot2D(X,Y,Alpha):
    RotX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RotY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RotX, RotY

def odesys(y, t, P, c, l, g, mu):  # функция системы уравнений
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = 1
    a12 = 0
    a21 = 0
    a22 = l + y[0]

    b1 = g*np.cos(y[1]) + (l+y[0])*y[3]**2 - ((c*g)/P)*y[0] - ((mu*g)/P)*y[2]
    b2 = - y[2]*y[3] - g*(l + y[0])*np.sin(y[1])

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy


# задаём все параметры
P = 10       # вес колечка
l = 0.5      # длина стержня
c = 20       # жёсткость пружины
g = 9.81     # скорость свободного падения
mu = 10       # параметр мю
l_0 = 0.25   # длина недеформированной пружины

t_fin = 20

t = np.linspace(0, 10, 1001) # создаём сетку по времени


# начальное состояние
phi0 = np.pi/10
dphi0 = 0.3
s0 = 0
ds0 = 0
y0 = [s0, phi0, ds0, dphi0]  # вектор начального состояния

Y = odeint(odesys, y0, t, (P, c, l, g, mu))

s = Y[:, 0]   # получили решение
phi = Y[:, 1]
ds = Y[:, 2]
dphi = Y[:, 3]

dds = np.array([odesys(yi, ti, P, c, l, g, mu)[2] for yi,ti in zip(Y,t)])
ddphi = np.array([odesys(yi, ti, P, c, l, g, mu)[3] for yi,ti in zip(Y,t)])

N = P*np.sin(phi) + P/g*((l + s)*ddphi + 2*ds*dphi)

fig_for_graphs = plt.figure(figsize=[13,7])  # построим их графики
ax_for_graphs = fig_for_graphs.add_subplot(2,2,1)
ax_for_graphs.plot(t,s,color='blue')
ax_for_graphs.set_title("s(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,2)
ax_for_graphs.plot(t,phi,color='red')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,3)
ax_for_graphs.plot(t,ds,color='green')
ax_for_graphs.set_title("s'(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,4)
ax_for_graphs.plot(t,dphi,color='black')
ax_for_graphs.set_title('phi\'(t)')
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)


fig_for_N = plt.figure(figsize=[13,7])  # построим их графики
ax_for_N = fig_for_N.add_subplot(1,1,1)
ax_for_N.plot(t,N,color='blue')
ax_for_N.set_title("N(t)")
ax_for_N.set(xlim=[0,t_fin])
ax_for_N.grid(True)


xA = l * np.sin(phi)                         # координаты конца стержня А
yA = -l * np.cos(phi)

xO = x0 = 0
yO = y0 = 0

xM = (l_0 + s/5) * np.sin(phi)               # координаты колечка М
yM = - (l_0 + s/5) * np.cos(phi)

xT = np.array([-0.09/2, 0, 0.09/2, -0.09/2]) # опора O
yT = np.array([0.05, 0, 0.05, 0.05])



n = 13
h = 0.03
xG = np.linspace(0,1,2*n+1)
yG = np.zeros(2*n+1)
ss = 0
for i in range(2*n+1):
    yG[i] = h*np.sin(ss)
    ss += np.pi/2

L_Spr = l_0+s/5

RotX_Spr, RotY_Spr = Rot2D(xG*L_Spr[0], yG, phi[0] - np.pi/2)



fig = plt.figure(figsize=[13, 9])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-1, 1], ylim=[-1, 1])

AO = ax.plot([xA[0], xO], [yA[0], yO], color=[0, 0, 0])[0]                  # стержень
A = ax.plot(xA[0], yA[0], 'o', color=[0, 0, 0])[0]                          # конец стержня
O = ax.plot(xO, yO, 'o', color=[0, 0, 0])[0]                                # точка О
T = ax.plot(xT,yT, color=[0, 0, 0])[0]                                      # опора О
M = ax.plot(xM[0], yM[0], 'o', color=[0, 0, 1])[0]                          # колечко М
Spring = ax.plot(RotX_Spr, RotY_Spr, color=[0, 0, 1])[0]             # пружина 


def kadr(i):
    A.set_data(xA[i], yA[i])
    O.set_data(xO, yO)
    AO.set_data([xA[i], xO], [yA[i], yO])
    M.set_data(xM[i], yM[i])
    RotX_Spr, RotY_Spr = Rot2D(xG*L_Spr[i], yG, phi[i] - np.pi/2)
    Spring.set_data(RotX_Spr, RotY_Spr)
    return [A, O, AO, M, Spring]



kino = FuncAnimation(fig, kadr,frames=len(t),interval=10)

plt.show()
