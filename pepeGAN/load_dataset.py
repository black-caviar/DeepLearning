import sys
import PIL
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE    

def mobilenet_preprocess(x,y):
    #x = x.cast(x, tf.float32)
    x /= 127.5
    x -= 1.
    return x,y

def load_256(x,y):
    x = tf.io.read_file(x)
    x = tf.io.decode_image(x, expand_animations=False)
    x = tf.image.resize(x, [256,256])
    return x,y 

def get_image_list(path):
    return tf.data.Dataset.list_files(path + '*', shuffle=False)
    
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

def construct_datasets(p, np):
    n_p = tf.data.experimental.cardinality(p).numpy()
    n_np = tf.data.experimental.cardinality(np).numpy()
    print('Pepe images:', n_p)
    print('Non-Pepe images:', n_np)
    
    p = p.map(lambda x: (x,1))
    np = np.map(lambda x: (x,0))
    p = p.shuffle(3000)
    np = np.shuffle(3000)
    
    pnp = tf.data.experimental.sample_from_datasets([p, np])
    n_pnp = n_p + n_np
    pnp_train = pnp.take(int(0.8*n_pnp))
    pnp_val = pnp.skip(int(0.8*n_pnp))
    
    pnp_train = pnp_train.map(load_256)
    pnp_val = pnp_val.map(load_256)
    # I don't think this matters 
    return pnp_train, pnp_val 

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    import matplotlib.pyplot as plt
    #data = load_directory(sys.argv[1])
    p = get_image_list(sys.argv[1])
    np = get_image_list(sys.argv[2])
    x,y = construct_datasets(p,np)
    #x = get_binary_dataset(sys.argv[1])
    i=0
    for f in y.take(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(f[0].numpy().astype("uint8"))
        #plt.imshow(f[0].numpy())
        print(f[1].numpy())
        plt.title(int(f[1].numpy()))
        i=i+1
    plt.show()
