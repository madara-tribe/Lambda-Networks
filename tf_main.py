from tensorflow_.resnet import create_model
from tensorflow_.tf_lambda_network import LambdaLayer

import tensorflow as tf

def load_model():
    H=W=256
    model = create_model(256, 255)
    model.summary()
    return model

def main():
    c = 1028
    lamdaheads = 4
    x = tf.random.normal(shape=(1, 16, 16, c))
    out = LambdaLayer(dim_k=c/lamdaheads, r=3, heads = lamdaheads, dim_out = c)(x)
    print(out.shape)
