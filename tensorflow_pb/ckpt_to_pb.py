import  tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    output_node_names = "output/biases"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        for op in graph.get_operations():
            print(op.name, op.values())

freeze_graph('D:/model/age/inception/checkpoint-14999','./model/checkpoint-14999.pb')