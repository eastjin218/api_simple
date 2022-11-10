from flask import Flask, request, jsonify

from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
from typing import Tuple, Dict, List
import tensorflow as tf
import numpy as np
import os
import sys
import io
sys.path.append('../')
from od_framework import framework

MODEL_NAME = 'efficientdet_d2_coco'
GPU_number1 = '1'
AD_MODEL_NAME = 'efficientdet_d2_coco_with_adv_car'

app = Flask(__name__)


@app.route('/')
def hello_flask():
    return "hello world"


@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


def input_image(image_path: str) -> np.array:
    image_data = Image.open(image_path)
    resize_image = np.array(image_data.resize(
        (960, 540), resample=Image.NEAREST))
    return resize_image


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


def load_model():
    normal_model = framework.get_object_detection_model(
        MODEL_NAME, GPU_number1)
    ad_model = framework.get_object_detection_model(AD_MODEL_NAME, GPU_number1)
    return normal_model, ad_model


@app.route('/anomaly_image', methods=['POST'])
def anomaly_image():
    # data_part
    # request file_path // httpie
    # >> http --timeout=300 -v POST http://127.0.0.1:5000/anomaly_image image_path='/home/dataset/virtual_tesla_attack/train/teslanew_t_0_0_0_l_2.jpg'
    # request_data = request.json
    # resize_image = input_image(request_data['image_path'])
    # fn = os.path.basename(request_data['image_path'])

    # request binary_img, file_name // file based
    request_data = request.files
    img_binary = request_data['img_file']
    fn = request_data['file_name']
    stream = io.BytesIO(img_binary.read())
    img = Image.open(stream)
    resize_image = np.array(img.resize((960, 540), resample=Image.NEAREST))

    # predict_part
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
    return jsonify(result_dict)


if __name__ == "__main__":
    app.run()
