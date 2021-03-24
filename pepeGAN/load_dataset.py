import sys
import PIL
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops

#keras_data_augmentation = tf.keras.Sequential([
#    layers.experimental.preprocessing.RandomFlip("horizontal"),
#    layers.experimental.preprocessing.RandomRotation(0.2),
#    layers.experimental.preprocessing.RandomCrop(224,224),
#])


def augment_data(x,y):       
    x = tf.image.random_crop(x, [224,224,3]) #random crop to 224
    x = tf.image.random_flip_left_right(x) #flip half of images
    #x = tf.image.random_saturation(x, 
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

def is_png(contents):
    substr = string_ops.substr(contents, 0, 8)
    return math_ops.equal(substr, b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A')

def is_gif(contents):
    substr = string_ops.substr(contents, 0, 6)
    return math_ops.Any(
        math_ops.equal(
            substr,
            [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']))

#OOM when allocating tensor with shape[100523,256,256,3]  

def load_256(x):
    d = tf.io.read_file(x)
    if tf.io.is_jpeg(d):
        img = tf.io.decode_jpeg(
            d,
            channels=3,
            try_recover_truncated=True,
            acceptable_fraction=0.9,
            dct_method="INTEGER_FAST")
    else:
        img = tf.io.decode_image(d, 3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 256.
    img = tf.image.resize(img, [256,256])
    return img

def dummy_load(x):
    return tf.zeros([256,256,3], dtype=tf.uint8) 

def get_image_list(path):
    return tf.data.Dataset.list_files(path + '*', shuffle=False)

def construct_datasets(p, np):
    p = p.map(lambda x: (x,1.))
    np = np.map(lambda x: (x,0.))
    
    pnp = tf.data.experimental.sample_from_datasets([p, np])
    #this can take a seed
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
