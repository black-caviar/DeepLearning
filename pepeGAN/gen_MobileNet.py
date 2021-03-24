import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet 

def MobileNet_binary(weights='imagenet', shape=(224,224,3), train_core=False):
    inputs = layers.Input(shape)
    norm = keras.layers.BatchNormalization(axis=[1,2])(inputs)
    reg = tf.keras.regularizers.l2(l2=0.001)
    core = keras.applications.MobileNetV2(
        alpha=1.0,
        include_top=False,
        weights=weights,
        input_tensor=None,
        pooling=None,
        classifier_activation='softmax',
    )
    x = core(norm, training=train_core)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer='random_normal',
        bias_initializer='zeros',
        kernel_regularizer=reg)(x)
    return keras.Model(inputs, outputs)

def test_net():
    return keras.Sequential([
        #layers.Input((256,256,3)),
        layers.Input((224,224,3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    
print("Hello")
model = MobileNet_binary()
model.summary()
model.save('MobileNet_bin')
