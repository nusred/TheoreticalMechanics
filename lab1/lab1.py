import numpy as np # работа с числами
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp # библиотека работы с символами (формулы)

t = sp.Symbol('t') # переменная t, как в математике
r, phi = 1+sp.sin(8*t), t+0.5*sp.sin(8*t)
x, y = r*sp.cos(phi), r*sp.sin(phi)

Vx, Vy = sp.diff(x,t), sp.diff(y,t) # Вектор скорости
Wx, Wy = sp.diff(Vx,t), sp.diff(Vy,t) # Вектор ускорения
v, W = sp.sqrt(Vx ** 2 + Vy ** 2), sp.sqrt(Wx ** 2 + Wy ** 2) # квадратный корень из составляющих по координатам

F_x, F_y = sp.lambdify(t,x), sp.lambdify(t,y) # функция координат
F_Vx, F_Vy = sp.lambdify(t,Vx), sp.lambdify(t,Vy) # функция скорости
F_Wx, F_Wy = sp.lambdify(t,Wx), sp.lambdify(t,Wy) # функция ускорения

t = np.linspace(0,10,1001) # t = массив

X, Y = F_x(t), F_y(t)
Vx, Vy = F_Vx(t), F_Vy(t)
Wx, Wy = F_Wx(t), F_Wy(t)

fig = plt.figure(figsize = [8,8]) # рисуем график, в скобках размеры окна
ax = fig.add_subplot(1,1,1)
ax.axis('equal') # чтобы единица по x была единицей по y
ax.set(xlim=[-7,7], ylim=[-7,7])

ax.plot(X,Y)# траектория
# анимация точки
P = ax.plot(X[0],Y[0], marker='o')[0] # поставили точку
kf = 0.3 # коэффициенты для корректного отображения

def TheMagicOfThtMovent(i):
   P.set_data(X[i], Y[i])
   VLine = ax.arrow(X[i], Y[i], kf * Vx[i], kf * Vy[i], width=0.03, color="red")  # Вектор скорости
   WLine = ax.arrow(X[i], Y[i], kf*0.01 * Wx[i], kf * Vy[i], width=0.03, color="green")  # Вектор ускорения

   CVector = ax.arrow(X[i], Y[i], - kf * ((Vy[i] * (Vx[i] ** 2 + Vy[i] ** 2)) / (Vx[i] * Wy[i] - Wx[i] * Vy[i])),
                      kf * ((Vx[i] * (Vx[i] ** 2 + Vy[i] ** 2)) / (Vx[i] * Wy[i] - Wx[i] * Vy[i])),
                      width=0.03, color="black")  # Вектор кривизны

   return P, VLine, WLine, CVector

kino = FuncAnimation(fig,TheMagicOfThtMovent, frames = len(t), interval = 20,blit = True)

plt.show()
