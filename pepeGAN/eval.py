#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import load_dataset as ld
import gen_MobileNet as mn
import argparse
from tqdm import tqdm
import numpy as np

def main():
    model = keras.models.load_model(FLAGS.weights)    

    data = ld.get_image_list(FLAGS.path)
 
    if FLAGS.pepe:
        fpepe = open(FLAGS.pepe, 'w')
    if FLAGS.not_pepe:
        fn_pepe = open(FLAGS.not_pepe, 'w')

    BATCH_SIZE = 32
        
    n_data = tf.data.experimental.cardinality(data).numpy()

    data = data.map(lambda x: (ld.load_256(x),x))
    data = data.map(ld.resize_224)
    data = data.map(ld.mobilenet_preprocess)
    data = data.apply(tf.data.experimental.ignore_errors())
    #data = data.map(lambda x,y: x)
    #Don't know where this is best located
    data = data.batch(BATCH_SIZE)
    with tqdm(total=n_data) as pbar:
        for x in data:
            y = model(x, training=False)
            #y = model.predict(x)
            #result = np.stack([x[1].numpy().squeeze(), y.squeeze()], axis=1)
            result = np.stack([x[1].numpy().squeeze(), y.numpy().squeeze()], axis=1)
            for e in result:
                fname = e[0].decode('UTF-8')
                if (e[1] > 0.5) and fpepe:
                    fpepe.write("%s \t %f\n" % (fname, e[1]))
                elif fn_pepe:
                    fn_pepe.write("%s \t %f\n" % (fname, e[1]))
            pbar.update(BATCH_SIZE)
            

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model file'
    )
    p.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to files to evaluate'
    )
    p.add_argument(
        '--pepe',
        type=str,
        required=False,
        help='Path to pepe output file'
    )
    p.add_argument(
        '--not-pepe',
        type=str,
        required=False,
        help='Path to not pepe output file'
    )
    FLAGS = p.parse_args()
    #print(FLAGS)
    main()
