import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# replace with custom loader 
#(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Convert pixels to floats 
#train_images, test_images = train_images / 255.0, test_images / 255.0

# use functional model because Keras says so
# who am I to say otherwise

inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="MyModel")
model.summary()

# this needs additional packages
keras.utils.plot_model(model, "cifar10_model.png", show_shapes=True)

#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])

#history = model.fit(train_images, train_labels, epochs=10, 
#                    validation_data=(test_images, test_labels), batch_size=1024)



#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
import LoadCIFAR
(x_train, y_train), (x_test, y_test), meta = LoadCIFAR.CIFAR_2D(*LoadCIFAR.load_CIFAR10())

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
#print(y_test)
# y_test is one hot encoded array

top5 = tf.keras.metrics.TopKCategoricalAccuracy(5)

#metrics are not the same as loss functions though loss functions may be
#used as metrics
model.compile(
    #optimizer=keras.optimizers.RMSprop(1e-3),
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[top5],
)
# We restrict the data to the first 1000 samples so as to limit execution time
# on Colab. Try to train on the entire dataset until convergence!
history = model.fit(x_train, y_train, batch_size=256, epochs=1, validation_split=0.2)
#history = model.fit(x_train, y_train, epochs=20, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
