import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def step_schedule(epoch):
    #this learning rate may actually be incorrect
    #base_lr = 1e-5 in solver prototype
    #step interval is also off...
    l = 1e-4
    #this is supposed to happen after epochs or batches?
    n = epoch - 300000
    if n > 0:
        return l/(2 << n//100000)
    else:
        return l
        
def EPE(y_true, y_pred):
    dist = tf.norm(y_pred - y_true, ord='euclidean', axis=2)
    return tf.reduce_mean(dist)


#there is a difference between the model as trained and as deployed
#The following is the model as deployed
#or at least most of it
def FlowNetS_deployed():
    #stride of 2 for each layer
    #relu after each layer
    inputs = keras.Input(shape=(384,512,6))
    #inputs = layers.Concatenate(axis=3)(inputs['img0'], inputs['img1'])
    #perform concaction in network 
    x = layers.Conv2D(64, 7, 2, padding='same', name='conv1', activation='relu')(inputs)
    c2out = layers.Conv2D(128, 5, 2, padding='same', name='conv2', activation='relu')(x)
    x = layers.Conv2D(256, 5, 2, padding='same', name='conv3', activation='relu')(c2out)
    c31out = layers.Conv2D(256, 3, 1, padding='same', name='conv3_1', activation='relu')(x)
    x = layers.Conv2D(512, 3, 2, padding='same', name='conv4', activation='relu')(c31out)
    c41out = layers.Conv2D(512, 3, 1, padding='same', name='conv4_1', activation='relu')(x)
    x = layers.Conv2D(512, 3, 2, padding='same', name='conv5', activation='relu')(c41out)
    c51out = layers.Conv2D(512, 3, 1, padding='same', name='conv5_1', activation='relu')(x)
    x = layers.Conv2D(1024, 3, 2, padding='same', name='conv6', activation='relu')(c51out)
    #add in extra c6_1 layer from release model
    c61out = layers.Conv2D(1024, 3, 1, padding='same', name='conv6_1', activation='relu')(x)

    #Refinement section
    #kernel size 4 instead of 5?
    decon5 = layers.Conv2DTranspose(512, 4, 2, padding='same', name='deconv5', activation='relu')(c61out)
    flow6 = layers.Conv2D(2, 3, 1, padding='same', name='convolution1')(c61out)
    flow6cup = layers.Conv2DTranspose(2, 4, 2, padding='same', name='upsample_flow6to5')(flow6)
    #make sure to check the order on those concats
    cat2 = layers.Concatenate(axis=3)([c51out,decon5,flow6cup])

    decon4 = layers.Conv2DTranspose(256, 4, 2, padding='same', name='deconv4', activation='relu')(cat2)
    flow5 = layers.Conv2D(2, 3, padding='same', name='convolution2')(cat2)
    flow5up = layers.Conv2DTranspose(2, 4, 2, padding='same', name='upsample_flow5to4')(flow5)
    #it may be worth building a custom layer for this as it repeats a few times
    cat3 = layers.Concatenate(axis=3)([c41out, decon4, flow5up])

    decon3 = layers.Conv2DTranspose(128, 4, 2, padding='same', name='deconv3', activation='relu')(cat3)
    flow4 = layers.Conv2D(2, 3, padding='same', name='convolution3')(cat3)
    flow4up = layers.Conv2DTranspose(2, 4, 2, padding='same', name='upsample_flow4to3')(flow4)
    cat4 = layers.Concatenate(axis=3)([c31out, decon3, flow4up])

    decon2 = layers.Conv2DTranspose(64, 4, 2, padding='same', name='deconv2', activation='relu')(cat4)
    flow3 = layers.Conv2D(2, 3, padding='same', name='convolution4')(cat4)
    flow3up = layers.Conv2DTranspose(2, 4, 2, padding='same', name='upsample_flow3to2')(flow3)
    cat5 = layers.Concatenate(axis=3)([c2out, decon2, flow3up])

    x = layers.Conv2D(2, 3, 1, padding='same', name='convolution5')(cat5)
    x = x*20; #why? because!
    #some more bullshit here
    #padding does nothing here right?
    #some magic interpolation here
    #convolution with constants for scaling purposes see actual model wtf
    x = layers.experimental.preprocessing.Resizing(384, 512, interpolation="bilinear", name='resample4')(x)
    outputs = layers.Conv2D(2, 1, 1, padding='valid', name='convolution6')(x)
    #384, 512 output
    return inputs, outputs


model = keras.Model(*FlowNetS_deployed(), name="FlowNetS")
model.summary()
keras.utils.plot_model(model, "FlowNetS_model.png", show_shapes=True)

optimizer = tf.keras.optimizers.Adam(1e-4)
#model.compile(loss=EPE

SAVE_PERIOD = 10

rate_callback = keras.callbacks.LearningRateScheduler(step_schedule)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model-{epoch%SAVE_PERIOD:04d}.hdf5',
    save_freq='epoch',
    period='SAVE_PERIOD',
    save_weights_only=True)

#validation data is special
callbacks = [rate_callback, checkpoint_callback]
#history = model.fit(x, y, batch_size=8, epochs=1, callbacks=[rate_callback])

#batch_size = 8
#mode.fit( yayaya, [rate_callback])

#output should be bilinearly interpolated to full resolution
#UpSampling2D is the function to use
