import tensorflow as tf
from model import  inception_v3

with tf.Session() as sess:
    AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    images = tf.placeholder(tf.float32, [None, 227, 227, 3])
    logits = inception_v3(len(AGE_LIST), images, 1, False)

    with tf.gfile.FastGFile('model/checkpoint-14999.pb', 'rb') as model_file:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
        softmax_output = tf.nn.softmax(logits)


    # [output_image] = tf.import_graph_def(graph_def,
    #                   input_map={'input_image': cv2.imread("D:/tmp/child-64x64x3-2.jpg")},
    #                   return_elements=['output_label:0'],
    #                   name='output')
    # sess = tf.Session()
    # label = sess.run(output_image)
    # print(label)


#
# # 读取图文件
# with tf.gfile.FastGFile('./model/checkpoint-14999.pb', 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     # We load the graph_def in the default graph
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(
#             graph_def,
#             input_map=None,
#             return_elements=None,
#             name="",
#             op_dict=None,
#             producer_op_list=None
#         )
#         with tf.Session() as sess:
#             # 根据名称返回tensor数据
#             inputs_vocab = graph.get_tensor_by_name('inputs_vocab:0')
#             feature_data_list = graph.get_tensor_by_name('inputs_feature_list:0')
#             sequence_length = graph.get_tensor_by_name('sequence_length:0')
#             max_length = graph.get_tensor_by_name('max_length:0')
#             # 准备测试数据(略)
#             # in_data = ...
#             # fea_data_list = ...
#             # length = ...
#             # max_len = ...
#             # feed 数据
#             feed_dict = {inputs_vocab.name: in_data,
#                          feature_data_list.name: fea_data_list,
#                          sequence_length.name: length,
#                          max_length.name: max_len}
#             # 计算结果
#             viterbi_sequence = graph.get_tensor_by_name('viterbi_sequence:0')
#             intent_prediction = graph.get_tensor_by_name('intent_prediction:0')
#             viterbi_sequence = sess.run(viterbi_sequence, feed_dict)
#             intent_prediction = sess.run(intent_prediction, feed_dict)
