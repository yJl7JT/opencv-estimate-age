from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from util.image_utils import image_batch
from tensorflow.python.framework import graph_util

RESIZE_FINAL = 227
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

with tf.Session() as sess:
    nlabels = len(AGE_LIST)
    from model import inception_v3
    images = tf.placeholder(tf.float32, [None, 227, 227, 3],name='input')
    logits = inception_v3(nlabels, images, 1, False)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    saver.restore(sess, 'D:\\model\\age\\inception\\checkpoint-14999')

    softmax_output = tf.nn.softmax(logits,name='softmax')

    image = image_batch("../test1.jpg")

    batch_results = sess.run(softmax_output,feed_dict={images:image.eval()})

    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]

    output /= batch_sz
    best = np.argmax(output)
    best_choice = (AGE_LIST[best], output[best])
    print('Guess @ 1 %s, prob = %.2f' % best_choice)


    # 生成pd模型
    sess.run(tf.global_variables_initializer())
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['softmax'])
    model_f = tf.gfile.FastGFile("./age.pb", mode="wb")
    model_f.write(output_graph_def.SerializeToString())


