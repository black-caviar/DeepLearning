import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 50 #num datapoints
M = 10 #num basis functions 
rate = 0.01 #adjustment rate
num_iter = 100 #num optimization iterations

#generate noisy sinusoidal data 
#x = tf.random.uniform([N], dtype=tf.float64)
x = tf.Variable(np.random.rand(N))
y = tf.math.sin(np.pi * 2 * x) + 0.1 * np.random.randn(N)

#generate clean data for plots 
clean_x = np.linspace(0,1,100)
clean_y = np.sin(np.pi * 2 * clean_x)

#generate initial state for basis functions 
params = {
    'w' : tf.Variable(np.zeros([M,1]) + 0.1),
    'b' : tf.Variable(np.zeros([M,1]) + 0.0),
    #'b' : tf.Variable(np.array([0.0]).reshape(-1,1), dtype=tf.float64),
    'mu' : tf.Variable(np.random.uniform(0,1,M).reshape(-1,1)),
    'sig': tf.Variable(np.zeros([M,1]) + 0.2)
    #'sig': tf.Variable(np.random.uniform(0,1,M).reshape(-1,1))
}
    
def y_h(param, x):
    acc = param['w'] * tf.math.exp(-((x-param['mu'])**2)/param['sig']**2) + param['b']
    return tf.math.reduce_sum(acc, axis=0)

def j(param, x, y):
    y_est = y_h(param, x)
    return (y-y_est)**2 / 2

#why am I doing loops instead of using matrices?
def plot_basis(M, p, x):
    for j in range(M):
        p_j = {}
        for i in p:
            p_j[i] = ((params[i]).numpy()[j]).reshape(-1,1)
        plt.plot(x, y_h(p_j, x))

#show initial basis functions 
plot_basis(M, params, clean_x)
plt.title("Initial Basis Functions")
plt.xlabel('x'); plt.ylabel('y', rotation=0); plt.xlim([0,1])
#plt.show()
plt.savefig('initial_basis.pdf', format='pdf')
plt.clf()

#view before optimization
fit_bad = y_h(params, clean_x)
plt.plot(clean_x, clean_y, label='Clean Sin')
plt.plot(clean_x, fit_bad, '--', label='Initial Fit')
plt.plot(x.numpy(), y.numpy(), '.', label='Data points')
plt.xlabel('x'); plt.ylabel('y', rotation=0); plt.xlim([0,1])
plt.title("Estimate Before Optimization")
plt.legend()
plt.show()
plt.savefig('before_opt.pdf', format='pdf')
plt.clf()

#gradient descent loop 
for i in range(num_iter):
    with tf.GradientTape() as tape:
        loss = j(params, x, y)
    grad = tape.gradient(loss, params)
    params['w'].assign(params['w'] - rate*grad['w'])
    params['b'].assign(params['b'] - rate*np.divide(params['b'].numpy(), grad['b'].numpy()))
    #scale because otherwise result is unreasonable
    params['mu'].assign(params['mu'] - rate*grad['mu'])
    params['sig'].assign(params['sig'] - rate*grad['sig'])

#plot the optimized result
y_fit = y_h(params, clean_x)
plt.plot(clean_x, clean_y, label='Clean Sin')
plt.plot(clean_x, y_fit, '--', label='Optimized Fit')
plt.plot(x.numpy(), y.numpy(), '.', label='Data points')
#plt.plot(clean_x, j(params, clean_x, clean_y))
plt.title("Estimate After Optimization")
plt.legend()
plt.xlabel('x'); plt.ylabel('y', rotation=0); plt.xlim([0,1])
plt.show()
plt.savefig('after_opt.pdf', format='pdf')
plt.clf()

plot_basis(M, params, clean_x)
plt.title("Basis Functions After Optimization")
plt.xlabel('x'); plt.ylabel('y', rotation=0); plt.xlim([0,1])
#plt.show()
plt.savefig('final_basis.pdf', format='pdf')
