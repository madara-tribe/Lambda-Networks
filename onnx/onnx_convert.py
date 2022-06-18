import os, sys
sys.path.append('../')
os.environ['TF_KERAS'] = '1'
import numpy as np
from keras2onnx import convert_keras
import onnx
from tensorflow.keras.models import Model
from UTKload import RACE_NUM_CLS
from train import ArcFace


OUTPUT_ONNX_MODEL_NAME = 'embedding_model.onnx'

def main(weight_path):
    arcface_ = ArcFace(train_path=None, val_path=None, num_race=RACE_NUM_CLS)
    m = arcface_.load_arcface_model(weights=None)
    m.load_weights(weight_path)
    embedding_model = Model(m.get_layer(index=0).input, m.get_layer(index=-5).output)
    embedding_model.summary()
    print(embedding_model.name)
    
    onnx_model = convert_keras(embedding_model, embedding_model.name)
    onnx.save(onnx_model, OUTPUT_ONNX_MODEL_NAME)
    print("success to output as "+OUTPUT_ONNX_MODEL_NAME)

if __name__=='__main__':
    weight_path = '../weights/ep40arcface_model_260x260.hdf5'
    main(weight_path)


