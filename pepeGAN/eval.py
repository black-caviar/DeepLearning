import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import load_dataset as ld
import gen_MobileNet as mn
import argparse
from tqdm import tqdm

def main():
    model = mn.MobileNet_binary(weights=None, train_core=False)
    checkpoint = tf.train.Checkpoint(net=model)
    checkpoint.restore(FLAGS.weights).expect_partial()

    #try: model.load_weights(FLAGS.weights)
    #except Exception as e:
    #    print(e)
    #    exit(1)

    try: data = ld.get_image_list(FLAGS.path)
    except Exception as e:
        print(e)
        exit(-1)

    if FLAGS.pepe:
        fpepe = open(FLAGS.pepe, 'a')
    if FLAGS.not_pepe:
        fn_pepe = open(FLAGS.not_pepe, 'a')

    BATCH_SIZE = 1
        
    n_data = tf.data.experimental.cardinality(data).numpy()

    data = data.map(lambda x: (ld.load_256(x),x))
    data = data.map(ld.resize_224)
    #ld.show_dataset(data)
    data = data.map(ld.mobilenet_preprocess)
    data = data.apply(tf.data.experimental.ignore_errors())
    #data = data.map(lambda x,y: x)
    #Don't know where this is best located
    data = data.batch(BATCH_SIZE)
    #with tqdm(total=n_data) as pbar:
    for x in data:
        y = model(x, training=False)
        #y = model(x)
        print(y)
            #print(element[1].numpy(), y)
            #fname = e[1].numpy().decode('UTF-8')
            #if (y > 0.5) and fpepe:
            #    fpepe.write("%s \t %f\n" % (fname, y))
            #elif fn_pepe:
            #    fn_pepe.write("%s \t %f\n" % (fname, y))
            #pbar.update(BATCH_SIZE)

    #
    #    

    #data = data.batch(32)
    
    #result = model.predict(data, verbose=1)
    #print(result)

    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights'
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
    print(FLAGS)
    main()
