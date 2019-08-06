from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from utils import ImageCoder, make_batch, FaceDetector

RESIZE_FINAL = 227
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

tf.app.flags.DEFINE_string('filename', '',
                           'File (Image) or File list (Text/No header TSV) to process')

FLAGS = tf.app.flags.FLAGS

def classify(sess, label_list, softmax_output, coder, images, image_file):

    print('Running file %s' % image_file)
    image_batch = make_batch(image_file, coder, True)
    batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
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
    return best_choice
         
def make_batch1(filename,coder):
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    image = coder.decode_jpeg(image_data)
    # image = ic.decode_jpeg(image_data)
    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    image_batch = tf.stack([crop])

def main(argv=None):
    with tf.Session() as sess:
        nlabels = len(AGE_LIST)
        from model import inception_v3
        images = tf.placeholder(tf.float32, [None, 227, 227, 3])
        logits = inception_v3(nlabels, images, 1, False)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        saver.restore(sess, 'D:\\model\\age\\inception\\checkpoint-14999')
        softmax_output = tf.nn.softmax(logits)
        coder = ImageCoder()

        image_data = tf.gfile.FastGFile("test1.jpg", 'rb').read()
        image = coder.decode_jpeg(image_data)
        crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
        image_batch = tf.stack([crop])

        batch_results = sess.run(softmax_output, feed_dict={images: image_batch.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (AGE_LIST[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)

        nlabels = len(AGE_LIST)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)

            print('Guess @ 2 %s, prob = %.2f' % (AGE_LIST[second_best], output[second_best]))

if __name__ == '__main__':
    tf.app.run()
