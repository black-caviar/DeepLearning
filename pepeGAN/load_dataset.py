import sys
import PIL
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_binary_dataset(path, batch=8):
    x = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode='binary',
        batch_size=batch,
        class_names=['not_pepe','pepe'],
    )
    return mobilenet_preprocess(x)

def mobilenet_preprocess(x):
    x = x.map(lambda x,y: (x / 127.5, y))
    x = x.map(lambda x,y: (x - 1., y))
    return x


def get_dataset(path):
    x = load_directory(path)
    x = x.map(lambda x: x / 127.5)
    x = x.map(lambda x: x - 1.)
    return x
    

def gen_dataset(path):
    list_ds = tf.data.Dataset.list_files(path + '*.*')
    data = list_ds.map(tf.io.read_file)
    return tf.cast(data, tf.float32)/255.0

def load_directory(path):
    #this returns integers 
    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode=None,
        #labels='inferred',
        batch_size=8,
        #label_mode=None,
        );

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    import matplotlib.pyplot as plt
    data = load_directory(sys.argv[1])
    x = get_binary_dataset(sys.argv[1])
    i=0
    for f in x.unbatch().take(9):
        ax = plt.subplot(3, 3, i + 1)
        #plt.imshow(f.numpy().astype("uint8"))
        plt.imshow((f[0].numpy()+1)/2)
        print(f[1].numpy())
        plt.title(f[1].numpy())
        i=i+1
    plt.show()
