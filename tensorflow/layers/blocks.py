from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np


def dw_conv(init, nb_filter, k):
    residual = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(init)
    residual = Conv2D(nb_filter * k, (1, 1), strides=(2, 2), padding='same', use_bias=False)(init)
    x = Conv2D(nb_filter * k, (1, 1), strides=(2, 2), padding='same', use_bias=False)(init)
    x = BatchNormalization()(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(init)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = tfa.activations.mish(x)
    x = Dropout(0.4)(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    return x

def res_block(init, nb_filter, k=1):
    #x = Activation('relu')(init)
    x = tfa.activations.mish(init)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = tfa.activations.mish(x)
    x = Dropout(0.4)(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Squeeze_excitation_layer(x)

    x = layers.add([init, x])
    return x


def Squeeze_excitation_layer(input_x):
    ratio = 4
    out_dim = int(np.shape(input_x)[-1])
    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = Dense(units=int(out_dim / ratio))(squeeze)
    #excitation = Activation('relu')(excitation)
    excitation = tfa.activations.mish(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = layers.Reshape([-1,1,out_dim])(excitation)
    scale = layers.multiply([input_x, excitation])

    return scale

def first_conv(inputs, nb_filter, i, k=1):
    #0
    x = Conv2D(nb_filter[i] *k, (3, 3), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = tfa.activations.mish(x)
    x = Conv2D(nb_filter[i] *k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tfa.activations.mish(x)
    return x
    
def extraction(x, nb_filter, i, k):
    x = dw_conv(x, nb_filter[i], k)
    x = dw_conv(x, nb_filter[i+1], k)
    x = dw_conv(x, nb_filter[i+2], k)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x

