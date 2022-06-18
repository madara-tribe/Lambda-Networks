from pathlib import Path
import os, sys
sys.path.append('../')
import glob, time
import onnxruntime
import onnx
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Model
from UTKload import UTKLoad, WEIGHT_DIR, WIDTH, HEIGHT, RACE_NUM_CLS, ID_RACE_MAP
from train import ArcFace
from metrics.cosin_metric import cosine_similarity
from metrics.padding_resize import img_padding



x = 100
y = 100

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_PLAIN,
               font_scale=5, thickness=2):
    text_color = (255, 255, 255)
    cv2.putText(image, label, point, font, font_scale,
            text_color, thickness, lineType=cv2.LINE_AA)


def load_embeding_model(weight):
    arcface_ = ArcFace(train_path=None, val_path=None, num_race=RACE_NUM_CLS)
    m = arcface_.load_arcface_model(weights=None)
    print('loading......')
    m.load_weights(weight)
    embeding_model = Model(m.get_layer(index=0).input, m.get_layer(index=-5).output)
    embeding_model.summary()
    return embeding_model
    
def get_hold_vector(model, path):
    image_dir = Path(path)
    X_test, vector_label = [], []
    for i, image_path in enumerate(image_dir.glob("*.jpg")):
        image_name = image_path.name
        y_label = image_name.split("_")[0]
        img = cv2.imread(str(image_path))
        vector_label.append(y_label)
        X_test.append(img)
    #load_ = UTKLoad(gamma=2.0)
    #hold_vector, vector_label = load_.load_data(path=path, img_size=HEIGHT)
    hold_vector = np.array(X_test, dtype='float32')/255
    hold_vector = model.predict(hold_vector, verbose=1)
    return hold_vector, vector_label

def onnx_inference(img_path, onnx_path, hold_vector, vector_label):
    img_name = img_path
    age, gender, race, _ = img_name.split("_")
    y_label = int(race)
    image = cv2.imread(img_path)
    cimg = image.copy()
    image = img_padding(image, desired_size=HEIGHT)
    image = image.astype(np.float32)/255
    image = np.expand_dims(image, 0)
    acc = 0
    start = time.time()
    ort = onnxruntime.InferenceSession(onnx_path)
    input_name = ort.get_inputs()[0].name
    query_vector = ort.run(None, {input_name: image})[0]
    similarity = cosine_similarity(query_vector, hold_vector)
    label_candidate = [vector_label[idx] for idx in np.argsort(similarity[0])[::-1][:20]]
    frequent_idx = np.argmax(np.bincount(label_candidate))
    if y_label==frequent_idx:
        print('right')
    else:
        print('wrong')
    print("ONNX Inference Latency is", (time.time() - start)*1000, "[ms]")
    draw_label(cimg, (x, y), ID_RACE_MAP[y_label])
    cv2.imwrite('onnx_arcface_predinction.png', cimg.astype(np.uint8))
    

if __name__=='__main__':
    img_path = str(sys.argv[1])
    onnx_path = str(sys.argv[2])
    vector_path = '../H'
    weight = '../weights/ep40arcface_model_260x260.hdf5'
    model = load_embeding_model(weight)
    hold_vector, vector_label = get_hold_vector(model, vector_path)
    onnx_inference(img_path, onnx_path, hold_vector, vector_label)



