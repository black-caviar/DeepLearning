import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import load_dataset as ld
import sys

def test_data(name, data):
    if data.get_shape() != (256,256,3):
        print(data.get_shape())
        print(name)

files = ld.get_image_list(sys.argv[1])
data = files.map(lambda x: (x,ld.load_256(x)))
#data.map(test_data)
#data.cache()
for e in data:
    test_data(*e)
    
