import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import load_dataset as ld

weights_dict = dict()
def load_weights_from_file(weight_file):
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()
    return weights_dict

def set_layer_weights(model, weights_dict):
    for layer in model.layers:
        if layer.name in weights_dict:
            print(layer.name)
            cur_dict = weights_dict[layer.name]
            current_layer_parameters = list()
            if layer.__class__.__name__ == "BatchNormalization":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
                current_layer_parameters.extend([cur_dict['mean'], cur_dict['var']])
            elif layer.__class__.__name__ == "Scale":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            elif layer.__class__.__name__ == "SeparableConv2D":
                current_layer_parameters = [cur_dict['depthwise_filter'], cur_dict['pointwise_filter']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            elif layer.__class__.__name__ == "Embedding":
                current_layer_parameters.append(cur_dict['weights'])
            elif layer.__class__.__name__ == "PReLU":
                gamma =  np.ones(list(layer.input_shape[1:]))*cur_dict['gamma']
                current_layer_parameters.append(gamma)
            else:
                # rot 
                if 'weights' in cur_dict:
                    current_layer_parameters = [cur_dict['weights']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(np.squeeze(cur_dict['bias']))
            model.get_layer(layer.name).set_weights(current_layer_parameters)
    return model

def step_schedule(epoch):
    #this learning rate may actually be incorrect
    #base_lr = 1e-5 in solver prototype
    #step interval is also off...
    l = 1e-4
    #this is supposed to happen after epochs or batches?
    n = epoch - 300000
    if n > 0:
        return l/(2 << n//100000)
    else:
        return l

def fast_schedule(epoch):
    l = 1e-5
    return l/(2 << epoch//5)
        
def EPE(y_true, y_pred):
    y_true = y_true * 0.05
    dim = y_pred.shape.as_list()[1:-1]
    #print(dim)
    if y_true.shape != y_pred.shape:
        #lets hope batching works correctly
        y_true = tf.image.resize(y_true, size=dim, method=tf.image.ResizeMethod.BILINEAR)
    dist = tf.norm(y_pred - y_true, ord='euclidean', axis=-1)
    return tf.reduce_mean(dist)

def EPE_Accuracy(y_true, y_pred):
    y_true = y_true * 0.5
    dist = tf.norm(y_pred - y_true, ord='euclidean', axis=-1)
    return tf.reduce_mean(dist)

def FlowNetS_deployed(weight_file = None, trainable = False):
    weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None
    #stride of 2 for each layer
    #relu after each layer
    #inputs = keras.Input(shape=(384,512,6))
    img1 = keras.Input(shape=(384,512,3))
    img2 = keras.Input(shape=(384,512,3))
    catimg = layers.Concatenate(axis=3)([img1,img2])
    #inputs = layers.Concatenate(axis=3)(inputs['img0'], inputs['img1'])
    #perform concaction in network 
    x = layers.Conv2D(64, 7, 2, padding='same', name='conv1', activation='relu')(catimg)
    c2out = layers.Conv2D(128, 5, 2, padding='same', name='conv2', activation='relu')(x)
    x = layers.Conv2D(256, 5, 2, padding='same', name='conv3', activation='relu')(c2out)
    c31out = layers.Conv2D(256, 3, 1, padding='same', name='conv3_1', activation='relu')(x)
    x = layers.Conv2D(512, 3, 2, padding='same', name='conv4', activation='relu')(c31out)
    c41out = layers.Conv2D(512, 3, 1, padding='same', name='conv4_1', activation='relu')(x)
    x = layers.Conv2D(512, 3, 2, padding='same', name='conv5', activation='relu')(c41out)
    c51out = layers.Conv2D(512, 3, 1, padding='same', name='conv5_1', activation='relu')(x)
    x = layers.Conv2D(1024, 3, 2, padding='same', name='conv6', activation='relu')(c51out)
    #add in extra c6_1 layer from release model
    c61out = layers.Conv2D(1024, 3, 1, padding='same', name='conv6_1', activation='relu')(x)

    #Refinement section
    #kernel size 4 instead of 5?
    decon5 = layers.Conv2DTranspose(512, 4, 2, padding='same', name='deconv5', activation='relu')(c61out)
    flow6 = layers.Conv2D(2, 3, 1, padding='same', name='Convolution1')(c61out)
    flow6cup = layers.Conv2DTranspose(2, 4, 2, padding='same', name='upsample_flow6to5')(flow6)
    #make sure to check the order on those concats
    cat2 = layers.Concatenate(axis=3)([c51out,decon5,flow6cup])

    decon4 = layers.Conv2DTranspose(256, 4, 2, padding='same', name='deconv4', activation='relu')(cat2)
    flow5 = layers.Conv2D(2, 3, padding='same', name='Convolution2')(cat2)
    flow5up = layers.Conv2DTranspose(2, 4, 2, padding='same', name='upsample_flow5to4')(flow5)
    #it may be worth building a custom layer for this as it repeats a few times
    cat3 = layers.Concatenate(axis=3)([c41out, decon4, flow5up])

    decon3 = layers.Conv2DTranspose(128, 4, 2, padding='same', name='deconv3', activation='relu')(cat3)
    flow4 = layers.Conv2D(2, 3, padding='same', name='Convolution3')(cat3)
    flow4up = layers.Conv2DTranspose(2, 4, 2, padding='same', name='upsample_flow4to3')(flow4)
    cat4 = layers.Concatenate(axis=3)([c31out, decon3, flow4up])

    decon2 = layers.Conv2DTranspose(64, 4, 2, padding='same', name='deconv2', activation='relu')(cat4)
    flow3 = layers.Conv2D(2, 3, padding='same', name='Convolution4')(cat4)
    flow3up = layers.Conv2DTranspose(2, 4, 2, padding='same', name='upsample_flow3to2')(flow3)
    cat5 = layers.Concatenate(axis=3)([c2out, decon2, flow3up])

    flow2 = layers.Conv2D(2, 3, 1, padding='same', name='Convolution5')(cat5)
    x = flow2*20; #why? because!
    #some more bullshit here
    #padding does nothing here right?
    #some magic interpolation here
    #convolution with constants for scaling purposes see actual model wtf
    #x = layers.experimental.preprocessing.Resizing(384, 512, interpolation="bilinear", name='resample4')(x)
    #hardcoded values bad
    flow_full = tf.image.resize(x, size=(384,512), method=tf.image.ResizeMethod.BILINEAR, name='flow_full')
    #I don't think this convolution does much
    #outputs = layers.Conv2D(2, 1, 1, padding='valid', name='Convolution6')(x)
    #384, 512 output

    #flow = keras.Input(shape=(384,512,2))
    #why is this
    #flow = flow * 0.05
    
    
    model = [];
    if trainable:
        model = Model(inputs = [img1,img2], outputs = [flow_full, flow2, flow3, flow4, flow5, flow6])
    else:
        model = Model(inputs = [img1,img2], outputs = [flow_full])

    if weights_dict != None:
        set_layer_weights(model, weights_dict)
    return model

def test(model):
    import flowiz as fz
    path = 'testfiles/0000000-'
    img1 = tf.expand_dims(plt.imread(path+'img0.ppm'), 0)
    img2 = tf.expand_dims(plt.imread(path+'img1.ppm'), 0)
    img1 = tf.cast(img1, tf.float32)/255.0
    img2 = tf.cast(img2, tf.float32)/255.0
    
    print(img1.shape, img2.shape)
    flow = model.predict([tf.reverse(img1,[-1]),tf.reverse(img2,[-1])])
    
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(img1.numpy().squeeze())
    plt.subplot(1,3,2)
    plt.imshow(img2.numpy().squeeze())
    plt.subplot(1,3,3)
    print(flow.shape)
    plt.imshow(fz.convert_from_flow(flow.squeeze()))
    plt.show()
    
if __name__ == '__main__':

    #model = FlowNetS_deployed()
    model = FlowNetS_deployed('checkpoints/trained_weights.npy', trainable=False)
    model.summary()
    keras.utils.plot_model(model, "FlowNetS_model.png", show_shapes=True)

    #data_valid = ld.get_dataset('FlyingChairs_release/tfrecord/fc_val.tfrecords', 4)
    #data_train = ld.get_dataset('FlyingChairs_release/tfrecord/fc_train.tfrecords', 4)
    
    test(model)

