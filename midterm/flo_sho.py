import glob
import flowiz as fz
import matplotlib.pyplot as plt
import IO

# Demo flo file viewer

path = 'FlyingChairs_release/tfrecord/fc_train.tfrecords'

num = 1000

img1 = sorted(glob.glob('FlyingChairs_release/data/*img1.ppm'))
img2 = sorted(glob.glob('FlyingChairs_release/data/*img2.ppm'))
flow = sorted(glob.glob('FlyingChairs_release/data/*.flo'))

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(plt.imread(img1[num]))
plt.subplot(1,3,2)
plt.imshow(plt.imread(img2[num]))

plt.subplot(1,3,3)
flow_img = fz.convert_from_file(flow[num])
plt.imshow(flow_img)
plt.show()
