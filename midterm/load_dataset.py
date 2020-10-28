import tensorflow as tf
#chair_train_dataset = tf.data.TFRecordDataset('FlyingChairs_release/tfrecord/fc_train.tfrecords')
#chair_val_dataset = tf.data.TFRecordDataset('FlyingChairs_release/tfrecord/fc_val.tfrecords')

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _parse_record(example):
    tfrecord_format = {
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'img1': tf.io.FixedLenFeature((), tf.string),
        'img2': tf.io.FixedLenFeature((), tf.string),
        'flow': tf.io.FixedLenFeature((393216), tf.float32),
    }
    example =  tf.io.parse_single_example(example, tfrecord_format)
    #print(example.keys())
    #these don't work for some fucking reason
    h = example['height']
    w = example['width']
    h = 384
    w = 512
    img1 = tf.io.parse_tensor(example['img1'], tf.uint8)
    img1 = tf.reshape(img1, [h,w,3])
    img2 = tf.io.parse_tensor(example['img2'], tf.uint8)
    img2 = tf.reshape(img2, [h,w,3])
    #print(img1.shape, img2.shape)
    flow = example['flow']
    flow = tf.reshape(flow, [h,w,2])
    #is there way to auto generate dict?
    #return img1, img2, flow
    return {'img1':img1, 'img2':img2, 'flow':flow}
    
def load_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename, compression_type='ZLIB')
    dataset = dataset.map(_parse_record, num_parallel_calls=1)
    return dataset

def get_dataset(filename, batch_size):
    dataset = load_dataset(filename)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def test_dataset(filename):
    import matplotlib.pyplot as plt
    import flowiz as fz
    data = get_dataset(filename, 1)
    elm = next(iter(data))

    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(elm['img1'].numpy().squeeze())
    plt.subplot(1,3,2)
    plt.imshow(elm['img2'].numpy().squeeze())
    plt.subplot(1,3,3)
    plt.imshow(fz.convert_from_flow(elm['flow'].numpy().squeeze()))
    plt.show()
    
'''
import tensorflow as tf
import load_dataset as ld
ld.test_dataset('FlyingChairs_release/tfrecord/fc_val.tfrecords')
'''
