import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import load_dataset as ld
import argparse
import datetime 
#load complete model
#save weights
#save complete model

AUTOTUNE = tf.data.experimental.AUTOTUNE    

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomCrop(224,224),
])

def augment_data(x,y):       
    x = tf.image.random_crop(x, [224,224,3]) #random crop to 224
    x = tf.image.random_flip_left_right(x) #flip half of images
    # there is no easy rotation in tf.image
    # but it can be done using PIL 
    return x,y

def main():
    try: model = keras.models.load_model(FLAGS.model)
    except Exception as e:
        print(e)
        exit(-1)

    if FLAGS.checkpoint:
        model.load_weights(FLAGS.checkpoint)

    try: pepe = ld.get_image_list(FLAGS.pepe)
    except Exception as e:
        print(e)
        exit(-1)

    try: not_pepe = ld.get_image_list(FLAGS.not_pepe)
    except Exception as e:
        print(e)
        exit(-1)

    n_p = tf.data.experimental.cardinality(pepe).numpy()
    n_np = tf.data.experimental.cardinality(not_pepe).numpy()
    n_samples = n_p + n_np
    print('Pepe images:', n_p)
    print('Non-Pepe images:', n_np)

    VAL_SPLIT = int(n_samples * 0.15)
    
    data = ld.construct_datasets(pepe, not_pepe)
    data = data.map(ld.mobilenet_preprocess)

    val = data.take(VAL_SPLIT)
    train = data.skip(VAL_SPLIT)
    
    train = train.map(augment_data)
    val = val.map(lambda x,y: (tf.image.resize(x, [224,224]),y))

    train = train.batch(32)
    val = val.batch(32)
    
    opt = keras.optimizers.Adam(learning_rate=0.01)
    loss = keras.losses.BinaryCrossentropy()

    c_path = 'checkpoints/cp-{epoch:04d}.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(
        c_path,
        monitor='val_binary_accuracy',
        mode='max',
        save_weights_only=True,
        save_best_only=True)

    log_dir = 'logdir/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callbacks=[tensorboard_callback, cp_callback]
    #callbacks=[tensorboard_callback]
    #callbacks=[]
    
    model.compile(optimizer=opt, loss=loss, metrics='binary_accuracy')
    model.fit(
        train,
        epochs=FLAGS.epochs,
        validation_data=val,
        callbacks=callbacks,
        verbose=1
    )
    #model.fit(pnp_train., epochs=10,
    #          validation_data=pnp_train.batch(4), callbacks=callbacks)
        
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--model',
        type=str,
        required=True,
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
