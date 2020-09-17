import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def pol2cart(r,th):
    x = r * np.cos(th)
    y = r * np.sin(th)
    return x, y

def gen_data(n):
    th = np.random.uniform(-np.pi, -9/2 * np.pi, size=[n,1])
    cl = np.random.randint(0,2,size=[n,1])
    return np.hstack(pol2cart(th*(2*cl-1), th)), cl

#generate a clean spiral
th = np.linspace(-np.pi, -9/2 * np.pi, num=100)
plt.plot(*pol2cart(th,th), 'b')
plt.plot(*pol2cart(-th,th), 'r')
#plt.show()

#th_red = np.random.uniform(-np.pi, -9/2 * np.pi, size=[200,1])
#th_blu = np.random.uniform(-np.pi, -9/2 * np.pi, size=[200,1])

#plt.plot(*pol2cart(th_blu,th_blu), 'bo') 
#plt.plot(*pol2cart(-th_red,th_red), 'ro')
#plt.show()

[x_test,y_test] = gen_data(400)
y_test = y_test.T[0]
plt.plot(x_test[y_test==0,0], x_test[y_test==0,1], 'ro')
plt.plot(x_test[y_test==1,0], x_test[y_test==1,1], 'bo')
plt.show()

num_data = 1000
thet = np.random.uniform(-np.pi, -9/2 * np.pi, size=[num_data,1])
#y = np.random.randint(0,2,size=[num_data,1])
#x = np.hstack(pol2cart(thet*(2*y-1),thet))
[x,y] = gen_data(num_data)
#print(x)
#plt.plot(x[:,0],x[:,1],'.')
#plt.show()

model = keras.Sequential(
    [
        layers.Dense(16, input_dim=2, activation="relu", name="hidden1"),
        layers.Dense(16, input_dim=2, activation="relu", name="hidden2"),
        layers.Dense(1,  activation='sigmoid', name="out"),
    ]
)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(x,y,epochs=200,batch_size=10,verbose=0)

yy = np.round(model.predict(x).T[0])
print(yy)

plt.plot(x[yy==0,0], x[yy==0,1], 'ro')
plt.plot(x[yy==1,0], x[yy==1,1], 'bo')
plt.show()

