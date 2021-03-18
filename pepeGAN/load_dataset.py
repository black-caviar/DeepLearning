import sys
import PIL
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE    

def mobilenet_preprocess(x,y):
    x /= 127.5
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
    p = p.map(lambda x: (x,1.))
    np = np.map(lambda x: (x,0.))
    pnp = tf.data.experimental.sample_from_datasets([p, np]) #this can take a seed
    pnp = pnp.shuffle(5000)
    return pnp.map(lambda x,y: (load_256(x),y), num_parallel_calls=AUTOTUNE)

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    import matplotlib.pyplot as plt
    p = get_image_list(sys.argv[1])
    np = get_image_list(sys.argv[2])
    data = construct_datasets(p,np)
    for i, d in enumerate(data.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        #plt.imshow(d[1].numpy().astype("uint8"))
        plt.imshow(d[0].numpy())
        #print(d[0])
        plt.title(int(d[1].numpy()))
        plt.axis('off')
    plt.show()
