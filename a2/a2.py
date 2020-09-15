import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

def pol2cart(r,th):
    x = r * np.cos(th)
    y = r * np.sin(th)
    return x, y

th = np.linspace(-np.pi, -9/2 * np.pi, num=100)

plt.plot(*pol2cart(th,th), 'b')
plt.plot(*pol2cart(-th,th), 'r')
#plt.show()

th_red = np.random.uniform(-np.pi, -9/2 * np.pi, size=200)
th_blu = np.random.uniform(-np.pi, -9/2 * np.pi, size=200)

plt.plot(*pol2cart(th_blu,th_blu), 'bo')
plt.plot(*pol2cart(-th_red,th_red), 'ro')
#plt.show()

# x = [400,2] containing x,y coordinates of all points
# w1 = [2,256] first neuron layer


# set up a 3 layer perceptron with 2 hidden layers



# 1,400 * [400,256] -> [1,256]
# [400,1] * [1,256] -> [400,256]



#x  1*784

#h1 784*128 -> 1x128
#h2 128*256 -> 1x256
#out 256*10 -> 1x10
#given a single image containing 784 datapoints

#biases are set to be vectors uniformly added to each neuron in layer


#i am operating on single data points containing an X and a Y
#however there are hundreds of data points

#so the "learning" iterates over the set of spiral points
#no need to vectorize them each

#x = 1x2
#h1 = 2x128 -> 1x128
#h2 = 128x256 -> 1x256
#out = 256x1 -> 1x1

num_features = 2
num_classes = 1
n_hidden_1 = 128
n_hidden_2 = 256

learning_rate = 0.01
num_epochs = 100
batch_size = 10

#random_normal = tf.initializers.RandomNormal(dtype=tf.float64)

weights = {
    'h1': tf.Variable(np.random.normal(size=[num_features, n_hidden_1])),
    'h2': tf.Variable(np.random.normal(size=[n_hidden_1, n_hidden_2])),
    'out': tf.Variable(np.random.normal(size=[n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float64)),
    'b2': tf.Variable(tf.zeros([n_hidden_2] ,dtype=tf.float64)),
    'out': tf.Variable(tf.zeros([num_classes] ,dtype=tf.float64))
}

def F(x):
    l1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    l1 = tf.nn.sigmoid(l1)
    #use relu here

    l2 = tf.add(tf.matmul(l1, weights['h2']), biases['b2'])
    l2 = tf.nn.sigmoid(l2)

    out = tf.add(tf.matmul(l2, weights['out']), biases['out'])
    return tf.nn.softmax(out)

def J(y, y_h):
    y_pred = tf.clip_by_value(y_h, 1e-9, 1.0)
    return tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(y_pred)))

optimizer = tf.optimizers.SGD(learning_rate)

def run_opt(x, y):
    with tf.GradientTape() as tape:
        pred = F(x)
        loss = J(pred,y)

    #tvar = weights#.values() + biases.values()
    
    grad = tape.gradient(loss, weights)
    #optimizer.apply_gradients(zip(grad, tvar))
    print(grad)
    weights['h1'] = weights['h1'] - learning_rate * grad['h1']
    weights['h2'] = weights['h2'] - learning_rate * grad['h2']
    weights['out'] = weights['out'] - learning_rate * grad['out']

for i in range(num_epochs):
    th = tf.Variable(np.random.uniform(-np.pi, -9/2 * np.pi, size=batch_size))
    borr = tf.Variable(np.random.randint(0,2,size=batch_size), dtype=tf.float64)
    [x,y] = pol2cart((2*borr-1)*th.numpy(),th.numpy())
    inp = np.concatenate((x[:,np.newaxis], y[:,np.newaxis]), axis=1)
    #can use unwrapping wow
    run_opt(inp,borr)
    if i % 10 == 0:
        print("fuckit")
