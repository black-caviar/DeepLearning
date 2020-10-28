import tensorflow as tf
from tensorflow import keras
import FlowNet as fn
import load_dataset as ld

#script to train the model


model = fn.FlowNetS_deployed('checkpoints/trained_weights.npy', trainable=True)
model.summary()
#keras.utils.plot_model(model, "FlowNetS_model.png", show_shapes=True)

optimizer = tf.keras.optimizers.Adam(1e-4)
loss_weights = [0.32, 0.08, 0.02, 0.01, 0.005]
loss = {
    'Convolution5': fn.EPE,
    'Convolution4': fn.EPE,
    'Convolution3': fn.EPE,
    'Convolution2': fn.EPE,
    'Convolution1': fn.EPE}
metrics = {'tf_op_layer_ResizeBilinear': fn.EPE}
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

SAVE_PERIOD = 1
rate_callback = keras.callbacks.LearningRateScheduler(fn.fast_schedule)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model-{epoch%SAVE_PERIOD:04d}.hdf5',
    save_freq='epoch',
#    period='SAVE_PERIOD',
    save_weights_only=True)

batch_size = 4

data_valid = ld.get_dataset('FlyingChairs_release/tfrecord/fc_val.tfrecords', batch_size)
data_train = ld.get_dataset('FlyingChairs_release/tfrecord/fc_train.tfrecords', batch_size)
    
callbacks = [rate_callback, checkpoint_callback]
#history = model.fit(x, y, batch_size=8, epochs=1, callbacks=[rate_callback])

history = model.fit(data_train, batch_size=batch_size, epochs=1, validation_data=data_valid, callbacks=callbacks)
