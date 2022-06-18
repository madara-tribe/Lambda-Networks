import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from .blocks import *
from .tf_lambda_network import LambdaLayer

def create_model(h, w, k=1, lr=1e-3):
    lambda_heads = 4
    inputs = Input(shape=(h, w, 3))
    i = 0
    nb_filter = [16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16]

    # 0
    x0 = first_conv(inputs, nb_filter, i, k=1) 
    i += 1

    #1
    x = dw_conv(x0, nb_filter[i], k)
    x = res_block(x, k, nb_filter[i])
    x1 = res_block(x, k, nb_filter[i])
    i += 1

    #2
    x = dw_conv(x1, nb_filter[i], k)
    x = res_block(x, k, nb_filter[i])
    x2 = res_block(x, k, nb_filter[i])
    i += 1

    #3
    x = dw_conv(x2, nb_filter[i], k)
    x = res_block(x, k, nb_filter[i])
    x3 = res_block(x, k, nb_filter[i])
    i += 1

    #4
    x = dw_conv(x3, nb_filter[i], k)
    x = res_block(x, k, nb_filter[i])
    x4 = res_block(x, k, nb_filter[i])

    b, g, f, c = x4.shape
    cx = LambdaLayer(dim_k=c/lambda_heads, r=3, heads=lambda_heads, dim_out=c)(x4)
    cx = GlobalAveragePooling2D()(cx)
    cx = BatchNormalization()(cx)
    cx = Dense(11, activation='softmax', name='color_logits')(cx)
    
    sx = LambdaLayer(dim_k=c/lambda_heads, r=3, heads=lambda_heads, dim_out=c)(x4)
    sx = GlobalAveragePooling2D()(sx)
    sx = BatchNormalization()(sx)
    sx = Dense(2, activation='softmax', name="shape_logits")(sx)
                  
    model = Model(inputs=inputs, outputs=[cx, sx])
    #--------------- center ------------
    return model
