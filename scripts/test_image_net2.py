import tensorflow as tf
from preprocessing import preprocessing_factory
from nets import nets_factory

from tensorflow.python.util import compat



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

def read_tensor_from_image_file(preprocessing_fn, file_name):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    #TODO: set fast_mode=False, add_image_summaries=False
    normalized = preprocessing_fn(image_reader, 299, 299)

    with tf.Session() as sess:
        return sess.run(normalized)

image_path = '/home/chasun/src/imagnet_validation_data/ILSVRC2012_val_00015060.JPEG'
t = read_tensor_from_image_file(
    preprocessing_fn,
    image_path)
print(t.shape)