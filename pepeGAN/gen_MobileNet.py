import os
# Disable GPU to avoid conflicting with other TF sessons
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet 
import matplotlib.pyplot as plt

def MobileNet_binary(weights='imagenet', shape=(224,224,3)):
    i = layers.Input([224, 224, 3], dtype = tf.uint8)
    #x = tf.cast(i, tf.float32)
    #x = mobilenet.preprocess_input(x)
    #x /= 127.5
    #x -= 1.
    #model = keras.Sequential();
    #model.add(layers.Input(
    #model.add(layers.Dense())
    # no resizing layer
    core = keras.applications.MobileNetV2(
        input_shape=shape, alpha=1.0,
        include_top=False, weights=weights, input_tensor=None, pooling=None,
        classes=1000, classifier_activation='softmax', 
    )
    x = core(i)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1,
        activation='sigmoid',
        kernel_initializer='random_normal',
        bias_initializer='zeros')(x)
    return keras.Model(inputs=[i], outputs=[x])
    
    
print("Hello")
model = MobileNet_binary()
model.summary()
model.save('MobileNet_bin')
