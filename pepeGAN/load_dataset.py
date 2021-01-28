import sys
import PIL
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

def gen_dataset(path):
    list_ds = tf.data.Dataset.list_files(path + '*.*')
    data = list_ds.map(tf.io.read_file)
    return tf.cast(data, tf.float32)/255.0

def pepe_256(path):
    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        #labels='inferred',
        #batch_size=1,
        label_mode=None,
        );

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    import matplotlib.pyplot as plt
    data = pepe_256(sys.argv[1])
    i=0
    for f in data.unbatch().take(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(f.numpy().astype("uint8"))
        i=i+1
    plt.show()
