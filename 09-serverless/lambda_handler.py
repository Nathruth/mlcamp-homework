# lambda_handler.py
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime as rt

MODEL_PATH = "hair_classifier_empty.onnx"
TARGET_SIZE = (200, 200)  # same preprocessing as homework 8


def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(TARGET_SIZE, Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, 0)  # batch dimension
    return arr


# Lambda handler
def predict(event, context=None):
    # event is a dict: {"image_url": "..."}
    url = event["image_url"]
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content))

    input_arr = preprocess_image(img)

    sess = rt.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    prob = sess.run([output_name], {input_name: input_arr})
    return float(prob[0][0][0])
