import tensorflow as tf

with tf.Session() as sess:
    with open('age1.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print(graph_def)
        softmax = tf.import_graph_def(graph_def, return_elements=['softmax:0'])
        # print(sess.run(softmax))

