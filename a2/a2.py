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

# set up a 3 layer perceptron with 2 hidden layers

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
n_hidden_1 = 16
n_hidden_2 = 16

learning_rate = 0.1
num_epochs = 1001
batch_size = 1

#random_normal = tf.initializers.RandomNormal(dtype=tf.float64)

weights = {
    'h1': tf.Variable(np.random.normal(size=[num_features, n_hidden_1])),
    'h2': tf.Variable(np.random.normal(size=[n_hidden_1, n_hidden_2])),
    'out_h': tf.Variable(np.random.normal(size=[n_hidden_2, num_classes])),

    'b1': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float64)),
    'b2': tf.Variable(tf.zeros([n_hidden_2] ,dtype=tf.float64)),
    'out_b': tf.Variable(tf.zeros([num_classes] ,dtype=tf.float64))
}
biases = {
    
}

def F(x):
    l1 = tf.add(tf.matmul(x, weights['h1']), weights['b1'])
    #l1 = tf.nn.leaky_relu(l1)
    l1 = tf.nn.relu(l1)
    #use relu here

    l2 = tf.add(tf.matmul(l1, weights['h2']), weights['b2'])
    #l2 = tf.nn.leaky_relu(l2)
    l2 = tf.nn.relu(l2)

    out = tf.add(tf.matmul(l2, weights['out_h']), weights['out_b'])
    return tf.nn.relu(out)

def J(y, y_h):
    #y_h being the predicted 
    #print('y', y, '\ny_h', y_h)
    
    y_pred = tf.clip_by_value(y_h, 1e-9, 1.0)
    #y_pred = y_h
    #return (y-y_h)**2/2
    return -y*tf.math.log(y_pred)

#optimizer = tf.optimizers.SGD(learning_rate)

def run_opt(x, y):
    with tf.GradientTape() as tape:
        tape.watch(weights)
        pred = F(x)
        loss = J(y,pred)
        #print(loss)

    grad = tape.gradient(loss, weights)
    #print(grad)
    for p in weights:
        weights[p] = weights[p] - learning_rate * grad[p]


th = np.random.uniform(-np.pi, -9/2 * np.pi, size=batch_size)
borr = tf.Variable(1*np.random.randint(0,2,size=batch_size)-0, dtype=tf.float64)
[x,y] = pol2cart((2*borr-1)*th,th)
inp = tf.Variable(np.concatenate((x[:,np.newaxis], y[:,np.newaxis]), axis=1))

#print(F(inp))
#print(borr[:,np.newaxis])
#exit()

loss_vec = []
  
for i in range(num_epochs):

    th = np.random.uniform(-np.pi, -9/2*np.pi, size=batch_size)
    borr = tf.Variable(1*np.random.randint(0,2,size=batch_size)-0, dtype=tf.float64)
    [x,y] = pol2cart((2*borr-1)*th,th)
    inp = tf.Variable(np.concatenate((x[:,np.newaxis], y[:,np.newaxis]), axis=1))
    for j in range(1):
        run_opt(inp,borr[:,np.newaxis])
    #print('iter:', i)
    if i % 10 == 0:
        print("fuckit", i)
        out = F(inp)
        ss = np.concatenate((out>0.5, borr[:,np.newaxis]), axis=1)
        print(ss)
        loss = J(out,borr).numpy()[0][0]
        loss_vec.append(loss)
        print('loss', loss)
        #print(weights)
        

plt.clf()
plt.plot(loss_vec)
#plt.semilogy(loss_vec)
plt.show()
#red['th'] = np.random.uniform(-np.pi, -9/2 * np.pi, size=batch_size/2)
#blu['th'] = np.random.uniform(-np.pi, -9/2 * np.pi, size=batch_size/2)

#[red['x'],red['y']] = pol2cart(-th,th)
#[blu['x'],blu['y']] = pol2cart(th,th)

#plt.clf()
#plt.plot(red['x'],red['y'],'.r')
#plt.plot(blu['x'],blu['y'],'.b')
#plt.show()

