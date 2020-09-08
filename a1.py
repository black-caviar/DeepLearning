import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 50

x = np.random.rand(N)
y = np.sin(np.pi * 2 * x) #+ 0.1 * np.random.randn(N)

M = 10

s_x = np.linspace(0,1,100)
s_y = np.sin(np.pi * 2 * s_x)
#replace with lambda 

plt.plot(s_x, s_y)
plt.plot(x, y, '.')
plt.show()


