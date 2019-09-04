import tensorflow as tf
from tensorflow.core.framework import graph_pb2

graph_def = graph_pb2.GraphDef()
#with open('inception_v3_2016_08_28_frozen.pb', "rb") as f:
#  graph_def.ParseFromString(f.read())
#print(graph_def)
with tf.Session() as sess:
  with open('inception_v3_2016_08_28_frozen.pb', "rb") as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def)
  train_writer = tf.summary.FileWriter("log",sess.graph)
  train_writer.close()
