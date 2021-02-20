import tensorflow as tf
from tensorflow import keras
import load_dataset as ld
import argparse 
#load complete model
#save weights
#save complete model

def main():
    try: model = keras.models.load_model(FLAGS.model)
    except Exception as e:
        print(e)
        exit(-1)

    try: pepe = ld.get_dataset(FLAGS.pepe)
    except Exception as e:
        print(e)
        exit(-1)

    try: not_pepe = ld.get_dataset(FLAGS.not_pepe)
    except Exception as e:
        print(e)
        exit(-1)
    print("Done loading")
    model.summary()

    pepe.map(lambda x: (x,1))
    not_pepe.map(lambda x: (x,0))
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    
    #model.fit(
        
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model'
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
