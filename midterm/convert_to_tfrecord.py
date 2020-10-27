
import argparse
import os
import sys
import numpy as np
from progressbar import ProgressBar, Percentage, Bar
from scipy.misc import imread
import tensorflow as tf

#this code is brazenly stolen. Sue me
# don't

TRAIN = 1
VAL = 2

def open_flo_file(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (w[0], h[0], 2))
