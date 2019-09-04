import tensorflow as tf
from preprocessing import preprocessing_factory
from nets import nets_factory

from tensorflow.python.util import compat

input_height=299
input_width=299
model_file = '/tmp/a.pb'


graph_def = tf.GraphDef()

with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())

#./bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=/tmp/a/graph.pb --input_binary --input_checkpoint=/data/testdata/tf_checkpoints/inception_v4.ckpt --output_node_names InceptionV4/Logits/Predictions --output_graph=/tmp/b.pb
#python3 -m tf2onnx.convert --inputs Placeholder:0 --outputs InceptionV4/Logits/Predictions:0 --input /tmp/b.pb --output 1.onnx --opset 10
model_name = 'inception_v4'
network_fn = nets_factory.get_network_fn(
    model_name,
    num_classes=1001,
    is_training=False)
image_size = network_fn.default_image_size
input_shape = [None, image_size, image_size, 3]

image_reader = tf.placeholder(tf.uint8, shape=[None, None, 3])
preprocessing_fn = preprocessing_factory.get_preprocessing('inception')
normalized = tf.expand_dims(preprocessing_fn(image_reader, input_height,input_width), 0)
network_fn(normalized)

saver = tf.train.Saver()
with tf.Session() as sess:
    #train_writer = tf.summary.FileWriter("/tmp/b",sess.graph)
    saver.restore(sess, '/data/testdata/tf_checkpoints/inception_v4.ckpt')
    tf.io.write_graph(sess.graph_def, '/tmp/a', 'graph.pb',as_text=False)