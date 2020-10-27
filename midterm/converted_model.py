import keras
from keras.models import Model
from keras import layers
import keras.backend as K
import numpy as np
from keras.layers.core import Lambda
import tensorflow as tf


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


def KitModel(weight_file = None):
    global weights_dict
    weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None
        
    data            = layers.Input(name = 'data', shape = (384, 512, 6,) )
    conv1_input     = layers.ZeroPadding2D(padding = ((3, 3), (3, 3)))(data)
    conv1           = convolution(weights_dict, name='conv1', input=conv1_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(7, 7), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU1           = layers.Activation(name='ReLU1', activation='relu')(conv1)
    conv2_input     = layers.ZeroPadding2D(padding = ((2, 2), (2, 2)))(ReLU1)
    conv2           = convolution(weights_dict, name='conv2', input=conv2_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(5, 5), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU2           = layers.Activation(name='ReLU2', activation='relu')(conv2)
    conv3_input     = layers.ZeroPadding2D(padding = ((2, 2), (2, 2)))(ReLU2)
    conv3           = convolution(weights_dict, name='conv3', input=conv3_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(5, 5), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU3           = layers.Activation(name='ReLU3', activation='relu')(conv3)
    conv3_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU3)
    conv3_1         = convolution(weights_dict, name='conv3_1', input=conv3_1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU4           = layers.Activation(name='ReLU4', activation='relu')(conv3_1)
    conv4_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU4)
    conv4           = convolution(weights_dict, name='conv4', input=conv4_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU5           = layers.Activation(name='ReLU5', activation='relu')(conv4)
    conv4_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU5)
    conv4_1         = convolution(weights_dict, name='conv4_1', input=conv4_1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU6           = layers.Activation(name='ReLU6', activation='relu')(conv4_1)
    conv5_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU6)
    conv5           = convolution(weights_dict, name='conv5', input=conv5_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU7           = layers.Activation(name='ReLU7', activation='relu')(conv5)
    conv5_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU7)
    conv5_1         = convolution(weights_dict, name='conv5_1', input=conv5_1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU8           = layers.Activation(name='ReLU8', activation='relu')(conv5_1)
    conv6_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU8)
    conv6           = convolution(weights_dict, name='conv6', input=conv6_input, group=1, conv_type='layers.Conv2D', filters=1024, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU9           = layers.Activation(name='ReLU9', activation='relu')(conv6)
    conv6_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU9)
    conv6_1         = convolution(weights_dict, name='conv6_1', input=conv6_1_input, group=1, conv_type='layers.Conv2D', filters=1024, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    ReLU10          = layers.Activation(name='ReLU10', activation='relu')(conv6_1)
    Convolution1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU10)
    Convolution1    = convolution(weights_dict, name='Convolution1', input=Convolution1_input, group=1, conv_type='layers.Conv2D', filters=2, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    #deconv5_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(ReLU10)
    deconv5_input   = layers.ZeroPadding2D(padding = (0))(ReLU10)
    deconv5         = convolution(weights_dict, name='deconv5', input=deconv5_input, group=1, conv_type='layers.Conv2DTranspose', filters=512, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='same', use_bias=True)
    #upsample_flow6to5_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Convolution1)
    upsample_flow6to5_input = layers.ZeroPadding2D(padding = (0))(Convolution1)
    upsample_flow6to5 = convolution(weights_dict, name='upsample_flow6to5', input=upsample_flow6to5_input, group=1, conv_type='layers.Conv2DTranspose', filters=2, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='same', use_bias=True)
    ReLU11          = layers.Activation(name='ReLU11', activation='relu')(deconv5)
    
    Concat2         = layers.concatenate(name = 'Concat2', inputs = [ReLU8, ReLU11, upsample_flow6to5])
    Convolution2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Concat2)
    Convolution2    = convolution(weights_dict, name='Convolution2', input=Convolution2_input, group=1, conv_type='layers.Conv2D', filters=2, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
    #deconv4_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Concat2)
    deconv4_input   = layers.ZeroPadding2D(padding = (0))(Concat2)
    deconv4         = convolution(weights_dict, name='deconv4', input=deconv4_input, group=1, conv_type='layers.Conv2DTranspose', filters=256, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='same', use_bias=True)
    #upsample_flow5to4_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Convolution2)
    upsample_flow5to4_input = layers.ZeroPadding2D(padding = (0))(Convolution2)
    upsample_flow5to4 = convolution(weights_dict, name='upsample_flow5to4', input=upsample_flow5to4_input, group=1, conv_type='layers.Conv2DTranspose', filters=2, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='same', use_bias=True)
    ReLU12          = layers.Activation(name='ReLU12', activation='relu')(deconv4)
    Concat3         = layers.concatenate(name = 'Concat3', inputs = [ReLU6, ReLU12, upsample_flow5to4])
    #Convolution3_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Concat3)
    Convolution3_input = layers.ZeroPadding2D(padding = (0))(Concat3)
    Convolution3    = convolution(weights_dict, name='Convolution3', input=Convolution3_input, group=1, conv_type='layers.Conv2D', filters=2, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=True)
    #deconv3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Concat3)
    deconv3_input   = layers.ZeroPadding2D(padding = (0))(Concat3)
    deconv3         = convolution(weights_dict, name='deconv3', input=deconv3_input, group=1, conv_type='layers.Conv2DTranspose', filters=128, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='same', use_bias=True)
    #upsample_flow4to3_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Convolution3)
    upsample_flow4to3_input = layers.ZeroPadding2D(padding = (0))(Convolution3)
    upsample_flow4to3 = convolution(weights_dict, name='upsample_flow4to3', input=upsample_flow4to3_input, group=1, conv_type='layers.Conv2DTranspose', filters=2, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='same', use_bias=True)
    ReLU13          = layers.Activation(name='ReLU13', activation='relu')(deconv3)
    Concat4         = layers.concatenate(name = 'Concat4', inputs = [ReLU4, ReLU13, upsample_flow4to3])
    #Convolution4_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Concat4)
    Convolution4_input = layers.ZeroPadding2D(padding = (0))(Concat4)
    Convolution4    = convolution(weights_dict, name='Convolution4', input=Convolution4_input, group=1, conv_type='layers.Conv2D', filters=2, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=True)
    
    #deconv2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Concat4)
    deconv2_input   = layers.ZeroPadding2D(padding = (0))(Concat4)
    deconv2         = convolution(weights_dict, name='deconv2', input=deconv2_input, group=1, conv_type='layers.Conv2DTranspose', filters=64, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='same', use_bias=True)
    #upsample_flow3to2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Convolution4)
    upsample_flow3to2_input = layers.ZeroPadding2D(padding = (0))(Convolution4)
    upsample_flow3to2 = convolution(weights_dict, name='upsample_flow3to2', input=upsample_flow3to2_input, group=1, conv_type='layers.Conv2DTranspose', filters=2, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='same', use_bias=True)
    ReLU14          = layers.Activation(name='ReLU14', activation='relu')(deconv2)
    Concat5         = layers.concatenate(name = 'Concat5', inputs = [ReLU2, ReLU14, upsample_flow3to2])
    #Convolution5_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(Concat5)
    Convolution5_input = layers.ZeroPadding2D(padding = (0))(Concat5)
    Convolution5    = convolution(weights_dict, name='Convolution5', input=Convolution5_input, group=1, conv_type='layers.Conv2D', filters=2, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=True)
    #Eltwise4        = my_add()([Convolution5, None])
    Eltwise4 = Convolution5 + 20
    model           = Model(inputs = [data], outputs = [Eltwise4])
    #model           = Model(inputs = [data], outputs = [Convolution5])
    if weights_dict != None:
        set_layer_weights(model, weights_dict)
    return model

class my_add(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(my_add, self).__init__(**kwargs)
    def call(self, inputs):
        res = inputs[0] + inputs[1]
        self.output_shapes = K.int_shape(res)
        return res
    
    def compute_output_shape(self, input_shape):
        return self.output_shapes


def convolution(weights_dict, name, input, group, conv_type, filters=None, **kwargs):
    if not conv_type.startswith('layer'):
        layer = keras.applications.mobilenet.DepthwiseConv2D(name=name, **kwargs)(input)
        return layer
    elif conv_type == 'layers.DepthwiseConv2D':
        layer = layers.DepthwiseConv2D(name=name, **kwargs)(input)
        return layer
    
    inp_filters = K.int_shape(input)[-1]
    inp_grouped_channels = int(inp_filters / group)
    out_grouped_channels = int(filters / group)
    group_list = []
    if group == 1:
        func = getattr(layers, conv_type.split('.')[-1])
        layer = func(name = name, filters = filters, **kwargs)(input)
        return layer
    weight_groups = list()
    if not weights_dict == None:
        w = np.array(weights_dict[name]['weights'])
        weight_groups = np.split(w, indices_or_sections=group, axis=-1)
    for c in range(group):
        x = layers.Lambda(lambda z: z[..., c * inp_grouped_channels:(c + 1) * inp_grouped_channels])(input)
        x = layers.Conv2D(name=name + "_" + str(c), filters=out_grouped_channels, **kwargs)(x)
        weights_dict[name + "_" + str(c)] = dict()
        weights_dict[name + "_" + str(c)]['weights'] = weight_groups[c]
        group_list.append(x)
    layer = layers.concatenate(group_list, axis = -1)
    if 'bias' in weights_dict[name]:
        b = K.variable(weights_dict[name]['bias'], name = name + "_bias")
        layer = layer + b
    return layer

if __name__ == '__main__':
     #load_weights_from_file('50351a4ff04c469e8db58de103982ac7.pb')
     mymodel = KitModel('checkpoints/trained_weights.npy')
     mymodel.summary()
