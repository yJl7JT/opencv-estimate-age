import tensorflow as tf
from utils import ImageCoder, make_batch, FaceDetector
import numpy as np
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
RESIZE_FINAL = 227


def image_batch(image_path):
    coder = ImageCoder()
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    image = coder.decode_jpeg(image_data)
    crop = tf.image.resize_images(image, (227, 227))
    image_batch = tf.stack([crop])
    # print(image_batch)
    return  image_batch

# aa = image_batch()
# print(aa)

def result(batch_results):
    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]

    output /= batch_sz
    best = np.argmax(output)
    best_choice = (AGE_LIST[best], output[best])
    print('Guess @ 1 %s, prob = %.2f' % best_choice)
