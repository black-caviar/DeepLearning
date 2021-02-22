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
    #x = tf.image.grayscale_to_rgb(x)
    if tf.shape(x)[2] != 3:
        print("Fuck you 1")
        x = tf.zeros([224,224,3])
        return x,y
        
    x = tf.image.random_crop(x, [224,224,3]) #random crop to 224
    x = tf.image.random_flip_left_right(x) #flip half of images
    # there is no easy rotation in tf.image
    # but it can be done using PIL 
    return x,y

def resize_image(x,y):
    if tf.shape(x)[2] != 3:
        print("Fuck you 2")
        x = tf.zeros([224,224,3])
        return x,y
    x = tf.image.resize(x, [224,224])
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

        
    print("Found image files")
    # perhaps the validation should occurr before shuffling?
    # I think so

    # I will have image augmentation of the validation set if I don't split it off now
    
    pnp_train, pnp_val = ld.construct_datasets(pepe, not_pepe)
    pnp_train = pnp_train.map(ld.mobilenet_preprocess)
    pnp_val = pnp_val.map(ld.mobilenet_preprocess)
    
    pnp_train = pnp_train.map(augment_data, num_parallel_calls=AUTOTUNE)
    pnp_val = pnp_val.map(resize_image)
    #pnp_train = pnp_train.cache()
    #preprocessing needs to be done every epoch so do it later?

    pnp_train = pnp_train.batch(32).prefetch(10)
    pnp_val = pnp_val.batch(32)
    # my pipeline is bonkers inefficient. Really need to try the layers api
    
    pnp_train.cache()
    pnp_val.cache()
    
    model.summary()
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.BinaryCrossentropy()

    c_path = 'checkpoints/cp-{epoch:04d}.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(c_path, save_weights_only=True)

    log_dir = 'logdir/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='2,5')

    #callbacks=[tensorboard_callback, cp_callback]
    callbacks=[tensorboard_callback]
    
    model.compile(optimizer=opt, loss=loss, metrics='binary_accuracy')
    model.fit(
        pnp_train,
        epochs=10,
        validation_data=pnp_val,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=4,
    )
    #model.fit(pnp_train.batch(32), epochs=10,
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
