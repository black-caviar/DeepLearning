import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import load_dataset as ld
import gen_MobileNet as mn
import argparse
import datetime 

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_data(pepe, not_pepe):
    fdata = ld.construct_datasets(pepe, not_pepe)
    data = fdata.map(lambda x,y: (ld.load_256(x),y), num_parallel_calls=AUTOTUNE) 
    data = data.map(ld.mobilenet_preprocess, num_parallel_calls=AUTOTUNE)
    return data

def main():
    if FLAGS.checkpoint:
        model = mn.MobileNet_binary(train_core=True)
        model.load_weights(FLAGS.checkpoint)
    elif FLAGS.model:
        model = keras.models.load_model(FLAGS.model)
    else:
        model = mn.MobileNet_binary(train_core=True)

    try: pepe = ld.get_image_list(FLAGS.pepe)
    except Exception as e:
        print(e)
        exit(-1)

    try: not_pepe = ld.get_image_list(FLAGS.not_pepe)
    except Exception as e:
        print(e)
        exit(-1)

    model.summary()

    n_p = tf.data.experimental.cardinality(pepe).numpy()
    n_np = tf.data.experimental.cardinality(not_pepe).numpy()
    n_samples = n_p + n_np
    print('Pepe images:', n_p)
    print('Non-Pepe images:', n_np)
    VAL_SPLIT = int(n_samples * 0.15)
    
    data = get_data(pepe, not_pepe)

    val = data.take(VAL_SPLIT)
    train = data.skip(VAL_SPLIT)

    train = train.map(ld.augment_data, num_parallel_calls=AUTOTUNE).batch(32)
    val = val.map(ld.resize_224, num_parallel_calls=AUTOTUNE).batch(32)

    sched = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        #decay_steps=10,
        decay_rate=0.70,
        staircase=True)
    opt = keras.optimizers.Adam(learning_rate=sched)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)

    #c_path = 'checkpoints/cp-{epoch:04d}.ckpt'
    c_path = 'checkpoints/checkpoint-test3.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(
        c_path,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=False,
        save_best_only=True)

    log_dir = 'logdir/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callbacks=[tensorboard_callback, cp_callback]
    #callbacks=[tensorboard_callback]
    #callbacks=[]

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy'])
    
    model.fit(
        train,
        epochs=FLAGS.epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=1
    )
        
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--model',
        type=str,
        required=False,
        help='Path to model'
    )
    p.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        help='Path to last model checkpoint'
    )
    p.add_argument(
        '--epochs',
        type=int,
        required=True,
        help='Number of epochs to run for'
    )
    p.add_argument(
        '--pepe',
        type=str,
        required=True,
        help='Path to pepe folder'
    )
    p.add_argument(
        '--not-pepe',
        type=str,
        required=True,
        help='Path to not pepe folder'
    )
    FLAGS = p.parse_args()
    print(FLAGS)
    main()
