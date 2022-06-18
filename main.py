from layers.resnet import create_model
from layers.tf_lambda_network import LambdaLayer
import tensorflow as tf
def load_model():
    H=W=256
    model = create_model(256, 255)
    model.summary()
    return model

def main():
    c = 256
    lamdaheads = 4
    x = tf.random.normal(shape=(1, 16, 16, 256))
    out = LambdaLayer(dim_k=c/lamdaheads, r=3, heads = lamdaheads, dim_out = c)(x)
    print(out.shape)
