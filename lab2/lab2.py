import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def Rot2D(X,Y,Alpha):
    RotX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RotY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RotX, RotY

# задаём все параметры
P = 10       # вес колечка
l = 0.5      # длина стержня
c = 20       # жёсткость пружины
g = 9.81     # скорость свободного падения
mu = 10       # параметр мю
l_0 = 0.25   # длина недеформированной пружины

t_fin = 20

t = np.linspace(0, 10, 1001) # создаём сетку по времени

phi = 0.2 + np.sin(4*t)
s = 0.05 + np.cos(8*t)

x = l * np.sin(np.sin(phi))
y = -l * np.cos(np.sin(phi))

xA = l * np.sin(phi)                         # координаты конца стержня А
yA = -l * np.cos(phi)

xO = 0
yO = 0

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
