import tensorflow as tf
from tensorflow import keras
import FlowNet as fn
import load_dataset as ld
import argparse
import random

#script to train the model


def train(opts):
    #model = fn.FlowNetS_deployed('checkpoints/trained_weights.npy', trainable=True)
    model = []
    if opts.starting_weights==None:
        model = fn.FlowNetS_deployed('checkpoints/trained_weights.npy', trainable=True)
    else:
        model = fn.FlowNetS_deployed(trainable=True)
        model.load_weights

    model.summary()
    #keras.utils.plot_model(model, "FlowNetS_model.png", show_shapes=True)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_weights = [0.32, 0.08, 0.02, 0.01, 0.005]
    loss = {
        'Convolution5': fn.EPE,
        'Convolution4': fn.EPE,
        'Convolution3': fn.EPE,
        'Convolution2': fn.EPE,
        'Convolution1': fn.EPE
    }
    metrics = {'tf_op_layer_ResizeBilinear': fn.EPE}
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
    
    SAVE_PERIOD = 1
    SESSION_ID = random.randint(0,9999)
    rate_callback = keras.callbacks.LearningRateScheduler(fn.fast_schedule)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model-{SESSION_ID:04d}-{epoch:04d}.hdf5',
        save_freq='epoch',
        #    period='SAVE_PERIOD',
        save_weights_only=True)
    history_callback = keras.callbacks.CSVLogger(opts.stats_output, append=True)

    callbacks = [rate_callback, checkpoint_callback, history_callback]
    batch_size = 4
    
    data_valid = ld.get_dataset('FlyingChairs_release/tfrecord/fc_val.tfrecords', batch_size)
    data_train = ld.get_dataset('FlyingChairs_release/tfrecord/fc_train.tfrecords', batch_size)
    
    #history = model.fit(x, y, batch_size=8, epochs=1, callbacks=[rate_callback])
    
    model.fit(data_train, batch_size=4, epochs=opts.epochs, validation_data=data_valid, callbacks=callbacks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--starting-weights',
        type=str,
        required=False,
        help='File of starting weights, default is pretrained from paper'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        required=True,
        help='Number of training epochs to take'
    )
    parser.add_argument(
        '--stats-output',
        type=str,
        required=True,
        help='Output file for stats'
    )
    flags = parser.parse_args()
    print(flags)
    train(flags)
