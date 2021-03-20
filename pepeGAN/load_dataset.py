import sys
import PIL
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

#keras_data_augmentation = tf.keras.Sequential([
#    layers.experimental.preprocessing.RandomFlip("horizontal"),
#    layers.experimental.preprocessing.RandomRotation(0.2),
#    layers.experimental.preprocessing.RandomCrop(224,224),
#])


def augment_data(x,y):       
    x = tf.image.random_crop(x, [224,224,3]) #random crop to 224
    x = tf.image.random_flip_left_right(x) #flip half of images
    # there is no easy rotation in tf.image
    # but it can be done using PIL 
    return x,y

def resize_224(x,y):
    x = tf.image.resize(x, [224,224])
    return x,y

def mobilenet_preprocess(x,y):
    #x /= 127.5
    x *= 2.
    x -= 1.
    return x,y

def load_256(x):
    x = tf.io.read_file(x)
    x = tf.io.decode_image(x, channels=3, expand_animations=False, dtype=tf.float32)
    x = tf.image.resize(x, [256,256])
    if x.get_shape() == (256,256,1):
        x = tf.image.grayscale_to_rgb(x)
    return x 

def get_image_list(path):
    return tf.data.Dataset.list_files(path + '*', shuffle=False)
    
def gen_dataset(path):
    list_ds = tf.data.Dataset.list_files(path + '*.*')
    data = list_ds.map(tf.io.read_file)
    return tf.cast(data, tf.float32)/255.0

def construct_datasets(p, np):
    #n_p = tf.data.experimental.cardinality(p).numpy()
    #n_np = tf.data.experimental.cardinality(np).numpy()
    #print(n_p)
    #print(n_np)

    p = p.map(lambda x: (x,1.))
    np = np.map(lambda x: (x,0.))
    
    pnp = tf.data.experimental.sample_from_datasets([p, np]) #this can take a seed
    pnp = pnp.shuffle(5000)
    return pnp

def show_dataset(data):
    import matplotlib.pyplot as plt
    for i, d in enumerate(data.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        print(d)
        plt.imshow(d[0].numpy())
        plt.title(int(d[1].numpy()))
        plt.axis('off')
    plt.show()

def dummy_gen():
    import numpy as np
    while True:
        l = np.random.randint(0,1)
        label = tf.convert_to_tensor(l, dtype=tf.float32)
        #label = tf.cast(tf.random.uniform([1], 0, 1, dtype=tf.int32), tf.float32)
        img = tf.fill([256,256,3], label)
        yield img, label



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    p = get_image_list(sys.argv[1])
    np = get_image_list(sys.argv[2])
    data = construct_datasets(p,np)
    data = data.map(lambda x,y: (load_256(x),y))
    show_dataset(data)
