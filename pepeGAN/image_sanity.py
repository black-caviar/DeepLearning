import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
from tensorflow import keras 
import load_dataset as ld
import sys

AUTOTUNE = tf.data.experimental.AUTOTUNE

class MyCallback(keras.callbacks.Callback):
    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

def test_data(name, data):
    if data.get_shape() != (256,256,3):
        print(data.get_shape())
        print(name)

files = ld.get_image_list(sys.argv[1])
data = files.map(ld.load_256, num_parallel_calls=AUTOTUNE)
#data = files.map(ld.dummy_load)
data = data.apply(tf.data.experimental.ignore_errors())#log_warning=True))
data = data.batch(32)
model = keras.Sequential([
    keras.layers.Input((256,256,3)),
])
#y = model.predict(data)#, callbacks=[MyCallback()])

for x in data:
    print(model(x).get_shape())


    
