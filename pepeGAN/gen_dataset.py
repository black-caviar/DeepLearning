import os
# Disable GPU to avoid conflicting with other TF sessons
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFile
import argparse
import glob
#import IO
import sys
from shutil import copyfile

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

def image_example(img):
  #shape = img.shape
  #print(img.size)
  feature = {
    'height': _int64_feature(img.height),
    'width': _int64_feature(img.width),
    'img': _bytes_feature(tf.io.serialize_tensor(np.asarray(img))),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))       

def convert_dataset(files):
  filename = os.path.join(FLAGS.out)
  if not FLAGS.basic:
    print('bad hello')
    writer_opt = tf.io.TFRecordOptions(compression_type='ZLIB')
    writer = tf.io.TFRecordWriter(filename, options=writer_opt)
    for f in files:
      img_file = os.path.join(FLAGS.data, f)
      try: 
        img_data = Image.open(img_file).convert('RGB')
        tf_example = image_example(img_data)
        writer.write(tf_example.SerializeToString())
        img_data.close()
      except IOError as e:
        print('Unable to open file:', img_file)
        print(e)
        continue
      writer.close()
  else:
    print('hello')
    os.makedirs(FLAGS.out)
    for i,f in enumerate(files, start=1):
      img_file = os.path.join(FLAGS.data, f)
      _,ext = os.path.splitext(f)
      output_file = os.path.join(FLAGS.out, '{:06d}'.format(i) + ext)
      #print(output_file)
      copyfile(img_file, output_file)

def main():
  ImageFile.LOAD_TRUNCATED_IMAGES = True
  flist = open(FLAGS.list, 'r')
  files = flist.read().split('\n')
  #files may not exist, just do dir dump
  print('Generating dataset with', len(files), 'elements')
  convert_dataset(files)
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data',
    type=str,
    required=True,
    help='Directory of images'
  )
  parser.add_argument(
    '--list',
    type=str,
    required=False,
    help='File containing list of images to use'
  )
  parser.add_argument(
    '--out',
    type=str,
    required=True,
    help='tfrecord output file'
  )
  parser.add_argument(
    '--basic',
    required=False,
    action='store_true',
    help='Do not create a tfrecord file, create a folder with images instead'
  )
  FLAGS = parser.parse_args()
  
  # Verify arguments are valid
  main()
    
