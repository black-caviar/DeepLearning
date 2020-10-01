import numpy as np

def unpickle(fname):
    import pickle
    with open(fname, 'rb') as f:
        return pickle.load(f, encoding='bytes')

def load_CIFAR10():
    path = 'cifar-10-batches-py/data_batch_'
    data = np.array([]).reshape(0,3072)
    labels = np.array([], dtype='int32')
    for i in range(1,6):
        batch = unpickle(path + str(i))
        data = np.concatenate((data, batch[b'data']), axis=0)
        labels = np.concatenate((labels, batch[b'labels']), axis=0)
        
    test = unpickle('cifar-10-batches-py/test_batch')
    meta = unpickle('cifar-10-batches-py/batches.meta')
    # for whatever reason test labels are regular array, need to be wrapped
    return (data, labels), (test[b'data'], np.array(test[b'labels']).astype('int32')), meta[b'label_names']
                           

def load_CIFAR100():
    test  = unpickle('cifar-100-python/test')
    train = unpickle('cifar-100-python/train')
    meta  = unpickle('cifar-100-python/meta')
    return (test[b'data'], test[b'fine_labels']), (train[b'data'], train[b'fine_labels']), meta[b'fine_label_names']
    
    test_d = test[b'data'], test[b'fine_labels']
    train_d = train[b'data'], train[b'fine_labels']
    meta_d = meta[b'fine_label_names']
    
    return train_d, test_d, meta_d

def CIFAR_2D(train,test,meta):
    (train_x,train_y) = train
    (test_x,test_y) = test
    train_x = train_x.reshape(-1,3,1024).transpose([0,2,1]).reshape(-1,32,32,3)
    test_x = test_x.reshape(-1,3,1024).transpose([0,2,1]).reshape(-1,32,32,3)
    return (train_x, train_y), (test_x, test_y), meta

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    _, (train_x,train_y), labels = CIFAR_2D(*load_CIFAR100())
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_x[i])
        plt.xlabel(labels[train_y[i]].decode('utf-8'))
    plt.suptitle('CIFAR-100 Train Sample')
    plt.show()
    _, (train_x,train_y), labels = CIFAR_2D(*load_CIFAR10())
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_x[i])
        plt.xlabel(labels[train_y[i]].decode('utf-8'))
    plt.suptitle('CIFAR-10 Train Sample')
    plt.show()
    
