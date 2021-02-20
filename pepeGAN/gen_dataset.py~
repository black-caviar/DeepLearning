import os
# Disable GPU to avoid conflicting with other TF sessons
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import glob
import IO
import sys

# Based on code by Sam Pepose

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(img1, img2, flow):
    #assume all are same shape, checks done 
    shape = img1.shape
    feature = {
        'height': _int64_feature(shape[0]),
        'width': _int64_feature(shape[1]),
        'img1': _bytes_feature(tf.io.serialize_tensor(img1.flatten())),
        'img2': _bytes_feature(tf.io.serialize_tensor(img2.flatten())),
        'flow': _float_feature(flow)
    }
    #haha what is this line
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Flo reader based on 
# https://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
def open_flo_file(filename):
    with open(filename, 'rb') as f:
        magic, = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w,h = np.fromfile(f, np.int32, count=2)
            #h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            #return np.resize(data, (w[0], h[0], 2))
            #print(data.size)
            return w, h, data

def open_ppm_file(filename):
    return np.asarray(Image.open(filename))
        

path = 'FlyingChairs_release/'

def convert_dataset(indices, name):
  filename = os.path.join(FLAGS.out, name + '.tfrecords')
  writer_opt = tf.io.TFRecordOptions(compression_type='ZLIB')
  writer = tf.io.TFRecordWriter(filename, options=writer_opt)
  for i in indices:
    img1_file = os.path.join(FLAGS.data_dir, '%05d_img1.ppm' % (i + 1))
    img2_file = os.path.join(FLAGS.data_dir, '%05d_img2.ppm' % (i + 1))
    flow_file = os.path.join(FLAGS.data_dir, '%05d_flow.flo' % (i + 1))

    flo_w, flo_h, flo_dat = open_flo_file(flow_file.strip())
    img1_dat = open_ppm_file(img1_file)
    img2_dat = open_ppm_file(img2_file)

    if img1_dat.shape != img2_dat.shape:
      print('error, image shape mismatch')
      print(i)
    if img1_dat.shape[0:2] != (flo_h,flo_w):
      print('error, flo shape mismatch')
      print(i)
      print(img1_dat.shape, (flo_h,flo_w))

    tf_example = image_example(img1_dat, img2_dat, flo_dat)
    writer.write(tf_example.SerializeToString())
  writer.close()

TRAIN = 1
VAL = 2
def main():
    # Load train/val split into arrays
    train_val_split = np.loadtxt(FLAGS.train_val_split)
    train_idxs = np.flatnonzero(train_val_split == TRAIN)
    val_idxs = np.flatnonzero(train_val_split == VAL)

    # Convert the train and val datasets into .tfrecords format
    print('Converting validation set')
    convert_dataset(val_idxs, 'fc_val')
    print('Converting training set')
    convert_dataset(train_idxs, 'fc_train')
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='Directory that includes all .ppm and .flo files in the dataset'
  )
  parser.add_argument(
    '--train_val_split',
    type=str,
    required=True,
    help='Path to text file with train-validation split (1-train, 2-validation)'
  )
  parser.add_argument(
    '--out',
    type=str,
    required=True,
    help='Directory for output .tfrecords files'
  )
  FLAGS = parser.parse_args()
  
  # Verify arguments are valid
  if not os.path.isdir(FLAGS.data_dir):
    raise ValueError('data_dir must exist and be a directory')
  if not os.path.isdir(FLAGS.out):
    raise ValueError('out must exist and be a directory')
  if not os.path.exists(FLAGS.train_val_split):
    raise ValueError('train_val_split must exist')
  main()
    
