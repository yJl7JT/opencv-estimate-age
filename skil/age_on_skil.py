# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:24:56 2019
SINGLE BATCH
@author: LohJZ
"""
# DONT DELETE
#import skil_client
#from skil_client.rest import ApiException
#from pprint import pprint
#import requests
#import tensorflow as tf
#from utils import ImageCoder
#
#image_file = 'test1.jpg'
#
#configuration = skil_client.Configuration()
#configuration.host = 'http://localhost:9008'
#configuration.username = 'admin'
#configuration.password = 'admin123'
#
#r = requests.post("http://localhost:9008/login", json={"userId": "admin", "password": "admin123"})
#token = r.json()['token']
#
#configuration.api_key['authorization'] = f'Bearer {token}'
#api_instance = skil_client.DefaultApi(skil_client.ApiClient(configuration))
#api_response = api_instance.predictimage("rudecarnie", "default", "rudecarnie", image='test1.jpg')

import skil_client
# from skil_client.rest import ApiException
from pprint import pprint
import requests
import tensorflow as tf
from utils import ImageCoder
import uuid
from tensorflow.python.platform import gfile

import numpy as np

from scipy import misc
# import matplotlib.pyplot as plt

from utils import ImageCoder
from data import inputs, standardize_image

RESIZE_FINAL = 227
image_file = 'D:/tmp/227/child-64x64x3-3.jpg'
label_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

def convert_indarray(np_array):
        """Convert a numpy array to `skil_client.INDArray` instance.

        # Arguments
            np_array: `numpy.ndarray` instance.

        # Returns
            `skil_client.INDArray` instance.
        """
        return skil_client.INDArray(
            ordering='c',
            shape=list(np_array.shape),
            data=np_array.reshape(-1).tolist()
        )
def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
    filename: string, path of the image file.
    Returns:
    boolean indicating if the image is a PNG.
    """
    return '.png' in filename

def make_multi_crop_batch(filename, coder):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    
    image = coder.decode_jpeg(image_data)

    crops = []
    print('Running multi-cropped image')
    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL
    print("h: ", h)
    print("w : ", w)
    print("hl: ", hl)
    print("wl : ", wl)

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(standardize_image(tf.image.flip_left_right(crop)))

    corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = standardize_image(tf.image.flip_left_right(cropped))
        crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)
    return image_batch

def make_single_image_batch(image_path, coder):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    image = coder.decode_jpeg(image_data)
    crop = tf.image.resize_images(image, (227,227))
    image_batch = tf.stack([crop])
    return image_batch


with tf.Session() as sess:
    coder = ImageCoder()
    image_batch = make_single_image_batch(image_file, coder)
    image_batch = image_batch.eval()

configuration = skil_client.Configuration()
configuration.host = 'http://192.168.1.128:9008'
configuration.username = 'admin'
configuration.password = '123456'

r = requests.post("http://192.168.1.128:9008/login", json={"userId": "admin", "password": "123456"})
token = r.json()['token']

configuration.api_key['authorization'] = f'Bearer {token}'
api_instance = skil_client.DefaultApi(skil_client.ApiClient(configuration))

data = [convert_indarray(image_batch)]
body_data = skil_client.MultiPredictRequest(
                id=str(uuid.uuid1()),
                needs_pre_processing=False,
                inputs=data
            )

response = api_instance.multipredict("age", "default", "outputgraphwithsoftmax", body=body_data )
response = response.to_dict()
output = response['outputs'][0]
probabilities = output['data']
probabilities = np.array(probabilities)
best_index = np.argmax(probabilities)
print("best class: " , best_index)
print("probability: ", probabilities[best_index])

best_choice = (label_list[best_index], probabilities[best_index])
print('best index is : ', best_index)
print('Guess @ 1 %s, prob = %.2f' % best_choice)

