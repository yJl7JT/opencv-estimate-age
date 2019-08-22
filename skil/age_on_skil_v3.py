"""
Created on Thu Aug  8 11:24:56 2019
MULTIPLE BATCH
@author: LohJZ
"""
# from skil_client.models.base64_nd_array_body import Base64NDArrayBody
import skil_client,cv2,base64
from skil_client.rest import ApiException
from pprint import pprint
import requests
import tensorflow as tf
from utils import ImageCoder
import tensorflow as tf # Default graph is initialized when the library is imported
import uuid
from tensorflow.python.platform import gfile
import time
import numpy as np

from scipy import misc
# import matplotlib.pyplot as plt

from utils import ImageCoder
from data import inputs, standardize_image

start = time.time()

RESIZE_FINAL = 227
image_file = 'D:/tmp/227/64x64x3-2-c.jpg'
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


def image_to_base64(image_np):
	image = cv2.imencode('.jpg',image_np)[1]
	image_code = str(base64.b64encode(image))[2:-1]
	return image_code


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
    image = image_to_base64(image)

    crops = []
    print('Running multi-cropped image')
    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

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
    image_batch = make_multi_crop_batch(image_file, coder)
    image_batch = image_batch.eval()

configuration = skil_client.Configuration()
configuration.host = 'http://192.168.1.128:9008'
configuration.username = 'admin'
configuration.password = '123456'

r = requests.post("http://192.168.1.128:9008/login", json={"userId": "admin", "password": "123456"})
token = r.json()['token']

configuration.api_key['authorization'] = f'Bearer {token}'
api_instance = skil_client.DefaultApi(skil_client.ApiClient(configuration))


list_ind_array = [[convert_indarray(np.expand_dims(image_batch[i,:,:,:], axis=0))] for i in range(12)]

batch_results = []
index = 0
for data in list_ind_array:
    print("getting response for batch image ", index)
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
    batch_results.append(probabilities)
    index+=1
    time.sleep(0.5) #prevent spamming

output = batch_results[0]    
batch_sz = len(batch_results)
arg_max_each_batch = []
for i in range(1, batch_sz):
    arg_max_each_batch.append(np.argmax(batch_results[i]))
for i in range(1, batch_sz):
    output = output + batch_results[i]

output /= batch_sz
print("Output: ", output)
best = np.argmax(output)
best_choice = (label_list[best], output[best])
print('best index is : ', best)
print('Guess @ 1 %s, prob = %.2f' % best_choice)

nlabels = len(label_list)
if nlabels > 2:
    output[best] = 0
    second_best = np.argmax(output)
    print('second best index is : ', second_best)
    print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))


end = time.time()
print("完成时间: %f s" % (end - start))
#
# label = "小人" if best < 3  else "大人"
#
# import cv2
# from PIL import ImageFont, ImageDraw, Image
# import numpy as np
#
# bk_img = cv2.imread(image_file)
# bk_img = cv2.resize(bk_img,(500,500))
# # 设置需要显示的字体
# fontpath = "font/simsun.ttc"   # 32为字体大小
# font = ImageFont.truetype(fontpath, 32)
# img_pil = Image.fromarray(bk_img)
# draw = ImageDraw.Draw(img_pil)
# # 绘制文字信息<br># (100,300/350)为字体的位置，(255,255,255)为白色，(0,0,0)为黑色
# draw.text((230, 50), label, font=font, fill=(0,0,255))
# # draw.text((100, 350), "你好", font=font, fill=(255, 255, 255))
# bk_img = np.array(img_pil)
#
# cv2.imshow(" ", bk_img)
# cv2.waitKey()