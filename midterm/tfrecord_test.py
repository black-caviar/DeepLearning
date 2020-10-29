import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import glob
import flowiz as fz
import matplotlib.pyplot as plt
import IO
import numpy as np

# Test validity of tfrecord conversion

path = 'FlyingChairs_release/tfrecord/fc_val.tfrecords'
#path = 'FlyingChairs_release/images.tfrecords'
feature = {
    'height': tf.io.FixedLenFeature((), tf.int64),
    'width': tf.io.FixedLenFeature((), tf.int64),
    'img1': tf.io.FixedLenFeature((), tf.string),
    'img2': tf.io.FixedLenFeature((), tf.string),
    #'flow': tf.io.FixedLenFeature((), tf.string),
    'flow': tf.io.FixedLenFeature((393216), tf.float32),
    }

def _parse_record(proto):
    return tf.io.parse_single_example(proto, feature)

data = tf.data.TFRecordDataset(path, compression_type='ZLIB')

y = []

for x in  data.take(1):
    y = _parse_record(x)
    print(y.keys())

    img1 = tf.io.parse_tensor(y['img1'], tf.uint8).numpy()
    plt.imshow(img1.reshape(384,512,3))
    plt.show()
    img2 = tf.io.parse_tensor(y['img2'], tf.uint8).numpy()
    plt.imshow(img2.reshape(384,512,3))
    plt.show()
    print(img1.shape, img2.shape)

    flow = np.frombuffer(y['flow'].numpy(), dtype=np.float32)
    print(flow.shape)
    plt.imshow(fz.convert_from_flow(flow.reshape(384,512,2)))
    plt.show()

    
    
