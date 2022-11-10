from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel

from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
from typing import Tuple, Dict, List
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('../')
from od_framework import framework

app = FastAPI()
MODEL_NAME = 'efficientdet_d2_coco'
GPU_number1 = '1'
AD_MODEL_NAME = 'efficientdet_d2_coco_with_adv_car'


@app.get('/')
def read_root():
    return {"hello":"world!!"}

def input_image(image_path: str) -> np.array:
    image_data = Image.open(image_path)
    resize_image = np.array(image_data.resize(
        (960, 540), resample=Image.NEAREST))
    return resize_image

def load_model():
    normal_model = framework.get_object_detection_model(MODEL_NAME, GPU_number1)
    ad_model = framework.get_object_detection_model(AD_MODEL_NAME, GPU_number1)
    return normal_model, ad_model

def check_predict(model1, model2,
                  # image_path:str,
                  resized_image: np.array,
                  score_threshold=.5) -> Tuple:
    # resized_image = input_image(image_path)
    boxes1, scores1, labels1, _ = model1.predict(resized_image)
    boxes2, scores2, labels2, _ = model2.predict(resized_image)
    score1_threshold = np.where(np.array(scores1) >= score_threshold)[0]
    score2_threshold = np.where(np.array(scores2) >= score_threshold)[0]
    label = model1.labels
    if score1_threshold.shape[0] == 0:
        print(' Normal model Not Detection anything!!')
        label1 = ''
    else:
        label1 = label[int(np.array(labels1)[score1_threshold][0])]['name']
    if score2_threshold.shape[0] == 0:
        print('Anomaly detector Not Detection anything!!')
        label2 = ''
    else:
        label2 = label[int(np.array(labels2)[score2_threshold][0])]['name']
    return (label1, label2)

class Image_data(BaseModel):
    image_path:str

@app.post('/predict')
async def anomaly_image(items: Image_data):
    '''
    curl -X 'POST' \
    'http://127.0.0.1:8000/predict' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "image_path": "/home/dataset/virtual_tesla_attack/train/teslanew_t_0_0_0_l_2.jpg"
    }'
    '''
    
    resize_image = input_image(dict(items)["image_path"])
    fn = os.path.basename(dict(items)['image_path'])

    normal_model, ad_model = load_model()
    normal_model_label, adv_model_label = check_predict(
        normal_model, ad_model, resize_image)

    result_dict = {}
    if normal_model_label != adv_model_label:
        result_dict['n_condition'] = '비정상'
    else:
        result_dict['n_condition'] = '정상'
    result_dict['id'] = str(fn)
    result_dict['label'] = normal_model_label
    result_dict['asum_label'] = adv_model_label
    return result_dict