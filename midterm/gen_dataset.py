import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import IO

def get_imagepair(file_path):
    print(file_path)
    img1 = Image.open(file_path[0])
    img2 = Image.open(file_path[1])
    return np.concatenate((img1,img2), axis=2)

def make_data(dset):   
    img = get_imagepair(dset[0:2])
    flow = IO.read(dset[2])
    return img, flow


path = 'FlyingChairs_release/data/'

img1 = sorted(glob.glob(path + '*img1.ppm'))
img2 = sorted(glob.glob(path + '*img2.ppm'))
flow = sorted(glob.glob(path + '*.flo'))

data = np.array((img1,img2,flow)).T
data = np.array([make_data(d) for d in data])
print(data)
#dataset = tf.data.Dataset.from_tensor_slices(data)
#dataset = dataset.map(make_data)
exit()



data = tf.data.Dataset.list_files(path + '*.ppm', shuffle=False)
for f in data.take(5):
    print(f)
    img = Image.open(f.numpy())
    plt.imshow(img)
    plt.show()


#def get_imagepair(id):
    
#def load_path(file_path):
    
