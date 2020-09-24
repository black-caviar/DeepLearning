import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
#from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
import MNISTload

mnist = tf.keras.datasets.mnist

(x_pool, y_pool), (x_test, y_test) = MNISTload.load_MNIST()
#convert to float
x_pool, x_test = x_pool.astype("float32") / 255.0, x_test.astype("float32") / 255.0

x_train, y_train = x_pool[:3*len(x_pool)//4], y_pool[:3*len(x_pool)//4]
x_valid, y_valid = x_pool[3*len(x_pool)//4:], y_pool[3*len(x_pool)//4:]

#try to get rid of these datasets
#can use Dataset.take Dataset.skip for some of these
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.flatten = Flatten() #just a convenience layer to reshape my inputs 
    self.d1 = Dense(18, activation='elu', kernel_regularizer=keras.regularizers.l2(0.0001))
    #dropout hidden layer
    self.drp1 = Dropout(0.01)
    #output layer fully connected
    self.d2 = Dense(10)
  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    x = self.drp1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.004, beta_1=0.9, beta_2=0.999)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

model.compile(optimizer=optimizer, loss=loss_object, metrics=[train_accuracy])
history = model.fit(x_train, y_train, batch_size=1024, epochs=50, validation_data=(x_valid,y_valid))
model.summary()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

#test set results
results = model.evaluate(x_test,y_test, batch_size=32)
print("loss and accuracy", results)

exit()

@tf.function
def step(x,y):
  with tf.GradientTape() as tape:
    #print(x,y)
    y_h = model(x, training=True)
    #print(y_h)
    loss = loss_object(y,y_h)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, y_h)

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  #for x,y in zip(x_train,y_train):
  #  print(x.shape)
  #  print(y)
  #  print(type(y))
  #  step(x,y.astype("float32"))
  
  for images, labels in train_ds:
    #print(images.shape)
    #print(labels.shape)
    #train_step(images, labels)
    step(images,labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))

#for test_images, test_labels in test_ds:
#    test_step(test_images, test_labels)
