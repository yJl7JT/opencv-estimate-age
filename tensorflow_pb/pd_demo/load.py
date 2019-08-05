import tensorflow as tf
with tf.Session() as sess:
    model_f = tf.gfile.FastGFile("../model/model.pb", mode='rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model_f.read())
    c = tf.import_graph_def(graph_def, return_elements=["add2:0"])
    c2 = tf.import_graph_def(graph_def, return_elements=["add3:0"])
    x, x2, c3 = tf.import_graph_def(graph_def, return_elements=["x:0", "x2:0", "add:0"])

    print(sess.run(c))
    print(sess.run(c2))
    print(sess.run(c3, feed_dict={x: 20, x2: 2}))



# addop = tf.add(x, x2, name='add')
# addop2 = tf.add(var1, var2, name='add2')
# addop3 = tf.add(var3, var2, name='add3')