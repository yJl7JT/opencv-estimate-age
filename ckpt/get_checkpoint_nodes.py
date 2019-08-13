import tensorflow as tf

with tf.Session() as sess:
  saver = tf.train.import_meta_graph('D:/model/age/inception/checkpoint-14999.meta')
  saver.restore(sess, tf.train.latest_checkpoint('D:/model/age/inception/'))

  #如名所言，以上是查看模型中的trainable variables；或者我们也可以查看模型中的所有tensor或者operations，如下：
  # gv = [v for v in tf.global_variables()]
  # for v in gv:
  #   print(v.name)

  '''
  上面通过global_variables()获得的与前trainable_variables类似，只是多了一些非trainable的变量，
  比如定义时指定为trainable=False的变量，或Optimizer相关的变量。
    下面则可以获得几乎所有的operations相关的tensor：
  '''
  # ops = [o for o in sess.graph.get_operations()]
  # for o in ops:
  #     print(o.name)

#tensorflow ckpt模型和pb模型获取节点名称
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader('D:/model/age/inception/checkpoint-14999')
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
'''
tensor_name:  OptimizeLoss/InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/weights/Momentum
tensor_name:  OptimizeLoss/InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/weights/Momentum
tensor_name:  Variable
tensor_name:  output/biases
'''