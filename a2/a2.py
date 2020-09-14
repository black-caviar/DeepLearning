import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

th = np.linspace(-np.pi, -9/2 * np.pi, num=50)
r_red = -th
r_blu = th

def x(r,th):
    return r * np.cos(th)

def y(r,th):
    return r * np.sin(th)

plt.plot(x(r_blu, th), y(r_blu, th), 'b')
plt.plot(x(r_red, th), y(r_red, th), 'r')
plt.show()

th_red = np.random.uniform(-np.pi, -9/2 * np.pi, size=200)
th_blu = np.random.uniform(-np.pi, -9/2 * np.pi, size=200)

x_red = x(-th_red, th_red)
y_red = y(-th_red, th_red)

x_blu = x(-th_blu, th_blu)
y_blu = y(-th_blu, th_blu)

plt.plot(x_red, y_red, 'ro')
plt.plot(x_blu, y_blu, 'bo')
plt.show()

n_hidden_1 = 256 
n_hidden_2 = 256
n_input = 400
n_classes = 1

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
