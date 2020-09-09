import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 50

x = tf.random.uniform([N], dtype=tf.float64)
#x = np.random.rand(N)
#print(x)
y = tf.math.sin(np.pi * 2 * x) + 0.1 * np.random.randn(N)
#y = np.sin(np.pi * 2 * x)
#print(y)

#clean sin 
sin_x = np.linspace(0,1,100)
sin_y = np.sin(np.pi * 2 * sin_x)
#replace with lambda 

plt.plot(sin_x, sin_y)
plt.plot(x.numpy(), y.numpy(), '.')
#plt.show()

M = 10

#b = tf.random.uniform([M, 1])
#u = tf.random.uniform([M, 1])
#q = tf.random.uniform([M, 1])
#w = tf.random.uniform([M, 1])

my_params = {
    'w' : tf.Variable(np.zeros([M,1]) + 0.1),
    'b' : tf.Variable(np.zeros([M,1]) + 0.0),
    #'b' : tf.Variable(np.array([0.0]).reshape(-1,1), dtype=tf.float64),
    'mu' : tf.Variable(np.random.uniform(0,1,M).reshape(-1,1)),
    'sig': tf.Variable(np.zeros([M,1]) + 0.1)
}

print(my_params)

def y_h(param, x):
    acc = param['w'] * tf.math.exp(-((x-param['mu'])**2)/param['sig']**2) + param['b']
    return tf.math.reduce_sum(acc, axis=0)

def j(param, x, y):
    y_est = y_h(param, x)
    return (y-y_est)**2 / 2

ewave = y_h(my_params, sin_x)
plt.plot(sin_x, ewave)
#plt.plot(x, j(my_params, x, y), '.')
plt.show()

rate = 0.01
num_iter = 10

for i in range(num_iter):
    with tf.GradientTape() as tape:
        loss = j(my_params, x, y)
    grads = tape.gradient(loss, my_params)
    print(i)
    print(grads['b'])
    my_params['w'].assign(my_params['w'].numpy() - rate*grads['w'].numpy())
    #my_params['b'].assign(my_params['b'].numpy() - 0.0001*grads['b'].numpy())
    my_params['mu'].assign(my_params['mu'].numpy() - rate*grads['mu'].numpy())
    my_params['sig'].assign(my_params['sig'].numpy() - rate*grads['sig'].numpy())
    #ewave = y_h(my_params, sin_x)
    #plt.plot(sin_x, ewave)
    #plt.plot(x, j(my_params, x, y), '.')
    #plt.plot(sin_x, sin_y)
    #plt.show()

print(my_params)
ewave = y_h(my_params, sin_x)
plt.plot(sin_x, sin_y)
plt.plot(sin_x, ewave)
plt.plot(x.numpy(), y.numpy(), '.')
plt.show()

for base in range(M):
    sparm = {}
    for x in my_params:
        #print("HELLO")
        #print(base, x)
        #print(my_params[x][base])
        sparm[x] = ((my_params[x]).numpy()[base]).reshape(-1,1)
    print(sparm)
    plt.plot(sin_x, y_h(sparm, sin_x))

plt.show()
