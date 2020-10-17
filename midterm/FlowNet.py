import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def Refinement():
    ...

def FlowNetS():
    #stride of 2 for each layer
    #relu after each layer
    inputs = keras.Input(shape=(384,512,6))
    x = layers.Conv2D(64, 7, 2, padding='same',  name='conv1')(inputs)
    c2out = layers.Conv2D(128, 5, 2, padding='same', name='conv2')(x)
    x = layers.Conv2D(256, 5, 2, padding='same', name='conv3')(c2out)
    c31out = layers.Conv2D(256, 3, padding='same', name='conv3_1')(x)
    x = layers.Conv2D(512, 3, 2, padding='same', name='conv4')(c31out)
    c41out = layers.Conv2D(512, 3, padding='same', name='conv4_1')(x)
    x = layers.Conv2D(512, 3, 2, padding='same', name='conv5')(c41out)
    c51out = layers.Conv2D(512, 3, padding='same', name='conv5_1')(x)
    x = layers.Conv2D(1024, 3, 2, padding='same', name='conv6')(c51out)

    #Refinement section
    #x = layers.Conv2D(512, 

    outputs = x
    return inputs, outputs

model = keras.Model(*FlowNetS(), name="FlowNetS")
model.summary()
keras.utils.plot_model(model, "FlowNetS_model.png", show_shapes=True)

#output should be bilinearly interpolated to full resolution
