import os, sys
import numpy as np
import cv2, json
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from DataLoder import DataLoad
from cfg import Cfg
from layers.resnet import create_model
 
def load_model(cfg, weights):
    model = create_model(cfg.H, cfg.W, k=1, lr=1e-3)    
    model.load_weights(weights)
    model.summary()
    return model
        
def evaluate(cfg, weight_path):
    loader = DataLoad(cfg)
    X_val, _, vc_label, vs_label = loader.meta_load(valid=True)
    X_val = np.array(X_val)
    model = load_model(cfg, weight_path)

    print('evaluating.....')
    sacc = cacc = 0
    
    pred_color, pred_shape = model.predict(X_val, verbose=1)
    for i, (c, s) in enumerate(zip(pred_color, pred_shape)):
        clabel, slabel = np.argmax(c), np.argmax(s)
        if clabel == vc_label[i]:
            cacc +=1
        if slabel==vs_label[i]:
            sacc +=1
    print("color, shape acc", cacc/len(X_val), sacc/len(X_val))
        
    
        
if __name__=='__main__':
    #idx = int(sys.argv[1])
    cfg = Cfg
    weight_path = "weights/arcface_model_10.hdf5"#.format(idx)
    evaluate(cfg, weight_path)
