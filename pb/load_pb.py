# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:19:00 2019

@author: LohJZ
"""
import tensorflow as tf  # Default graph is initialized when the library is imported
import os
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import cv2
from utils import ImageCoder
from data import inputs, standardize_image

RESIZE_FINAL = 227
image_file = '../old.jpg'
label_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


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

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(standardize_image(tf.image.flip_left_right(crop)))

    corners = [(0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl / 2), int(wl / 2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = standardize_image(tf.image.flip_left_right(cropped))
        crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)
    return image_batch


with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        print("load Graph")

        with gfile.FastGFile("age1.pb", "rb") as f:

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="",
                                op_dict=None, producer_op_list=None)

            l_input = graph.get_tensor_by_name("input:0")
            l_output = graph.get_tensor_by_name("softmax:0")

            print("Shape of Input: ", tf.shape(l_input))

            tf.global_variables_initializer()
            coder = ImageCoder()

            # create batch image
            image_batch = make_multi_crop_batch(image_file, coder)
            image_batch = image_batch.eval()

            # run session
            batch_results = sess.run(l_output, feed_dict={l_input: image_batch})

            # analyze output
            output = batch_results[0]
            batch_sz = batch_results.shape[0]

            for i in range(1, batch_sz):
                output = output + batch_results[i]

            output /= batch_sz
            best = np.argmax(output)
            best_choice = (label_list[best], output[best])
            print('Guess @ 1 %s, prob = %.2f' % best_choice)

            nlabels = len(label_list)
            if nlabels > 2:
                output[best] = 0
                second_best = np.argmax(output)
                print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

