import tensorflow as tf,numpy as np
from util.image_utils import image_batch

AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

saver = tf.train.import_meta_graph("ckpt/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "ckpt/model.ckpt")

    image = image_batch("../test1.jpg")

    softmax = tf.get_default_graph().get_tensor_by_name("softmax:0")
    input = tf.get_default_graph().get_tensor_by_name("input:0")

    init = tf.global_variables_initializer()

    batch_results = sess.run(softmax,feed_dict={input:image.eval()})
    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]

    output /= batch_sz
    best = np.argmax(output)
    best_choice = (AGE_LIST[best], output[best])
    print('Guess @ 1 %s, prob = %.2f' % best_choice)

