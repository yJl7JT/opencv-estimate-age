import tensorflow as tf
from tensorflow.python.framework import graph_util
var1 = tf.Variable(1.0, dtype=tf.float32, name='v1')
var2 = tf.Variable(2.0, dtype=tf.float32, name='v2')
var3 = tf.Variable(2.0, dtype=tf.float32, name='v3')
x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
x2 = tf.placeholder(dtype=tf.float32, shape=None, name='x2')
addop = tf.add(x, x2, name='add')
addop2 = tf.add(var1, var2, name='add2')
addop3 = tf.add(var3, var2, name='add3')
initop = tf.global_variables_initializer()
model_path = '../model/model.pb'
with tf.Session() as sess:
    sess.run(initop)
    print(sess.run(addop, feed_dict={x: 12, x2: 23}))
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['add', 'add2', 'add3'])
    # 将计算图写入到模型文件中
    model_f = tf.gfile.FastGFile(model_path, mode="wb")
    model_f.write(output_graph_def.SerializeToString())
