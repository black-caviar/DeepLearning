import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt

# use functional model because Keras says so
# who am I to say otherwise

# Input constructs tensor
def model1():
    inputs = keras.Input(shape=(32, 32, 3), name="img")
    #x = layers.Dropout(0.2)(inputs)
    x = layers.Conv2D(32, 3, activation="elu", bias_regularizer=regularizers.l2(1e-3))(inputs)
    x = layers.Conv2D(64, 3, activation="elu")(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation="elu", padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation="elu", padding="same", bias_regularizer=regularizers.l2(1e-4))(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation="elu", padding="same")(block_2_output)
    x = layers.Conv2D(64, 3, activation="elu", padding="same", bias_regularizer=regularizers.l2(1e-4))(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation="elu", bias_regularizer=regularizers.l2(1e-4))(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    #x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    return inputs, outputs

def model2():
    inputs = keras.Input(shape=(32, 32, 3), name="img")
    x = layers.Conv2D(3, 9, strides=1, activation="relu", padding='valid')(inputs)
    #there may be a pooling layer before the first conv
    x = layers.MaxPooling2D((2,2), strides=2)(x)

    x = layers.Conv2D(3, 5, strides=1, activation="relu", padding='same')(x)
    x = layers.MaxPooling2D((2,2), strides=2)(x)

    x = layers.Conv2D(3, 5, strides=1, activation="relu", padding='same')(x)
    x = layers.MaxPooling2D((2,2), strides=2)(x)

    x = layers.Conv2D(3, 5, strides=1, activation="relu", padding='same')(x)
    x = layers.MaxPooling2D((2,2), strides=2)(x)

    x = layers.Dense(384, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    return inputs, outputs
    
def model3():
    inputs = keras.Input(shape=(32, 32, 3), name="img")
    #add dropout here
    x = layers.Dropout(0.2)(inputs)
    x=inputs
    x = layers.Conv2D(96, 5, activation="relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, 5, activation="relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, 3, activation="relu")(x)
    x = layers.Conv2D(192, 1, activation="relu")(x)
    x = layers.Conv2D(10, 1, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10)(x)
    return inputs, outputs

model = keras.Model(*model1(), name="MyModel")
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

top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5_acc')
#accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='acc')

opt = keras.optimizers.Adam()
#0.01

# metrics are not the same as loss functions though loss functions may be
# used as metrics
model.compile(
    #optimizer=keras.optimizers.RMSprop(1e-3),
    optimizer=opt,
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[top5, 'acc'],
)

history = model.fit(x_train, y_train, batch_size=256, epochs=30, validation_split=0.2)
#history = model.fit(x_train, y_train, epochs=20, validation_split=0.2)

model.save('cifar10_model')

model.evaluate(x_test, y_test, verbose=2)

#print(history.history.keys())

plt.clf()
plt.plot(history.history['acc'], label='Training Acc')
plt.plot(history.history['val_acc'], label='Validation Acc')
plt.plot(history.history['top5_acc'], label='Training Acc 5')
plt.plot(history.history['val_top5_acc'], label='Validation Acc 5')
plt.legend()
plt.title("Accuracy over time")
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.savefig('cifar10_history.pdf', format='pdf')
plt.show()
