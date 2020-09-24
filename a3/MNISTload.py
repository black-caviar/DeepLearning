import numpy as np
import matplotlib.pyplot as plt

testimgfile = open('t10k-images-idx3-ubyte', 'rb')
testlabelfile = open('t10k-labels-idx1-ubyte', 'rb')

trainimgfile = open('train-images-idx3-ubyte', 'rb')
trainlabelfile = open('train-labels-idx1-ubyte', 'rb')

def get_images(f):
    if (int.from_bytes(f.read(4), 'big') != 2051):
        print("Not an image file")
        return None
    else:
        num_images = int.from_bytes(f.read(4), 'big')
        #print(num_images)
        
        dim = (int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big'))
        pixels = np.fromfile(f, np.uint8, num_images*dim[0]*dim[1], '', 0)
        images = pixels.reshape([num_images,28,28])
        return images

def get_labels(f):
    if (int.from_bytes(f.read(4), 'big') != 2049):
        print("Not a label file")
        return None
    else:
        num_labels = int.from_bytes(f.read(4), 'big')
        #print(num_labels)
        #offset here is 0 as I have already moved the file index
        labelvec = np.fromfile(f, np.uint8, num_labels, "", 0)
        return labelvec

def load_MNIST():
    x_train = get_images(trainimgfile)
    y_train = get_labels(trainlabelfile)
    x_test = get_images(testimgfile)
    y_test = get_labels(testlabelfile)
    return (x_train,y_train),(x_test,y_test)
    
if __name__ == '__main__':
    _, (x,y) = load_MNIST()
    #y = get_labels(testlabelfile)
    print(y[0])
    #x = get_images(testimgfile)
    plt.imshow(x[0,:,:])
    plt.show()

