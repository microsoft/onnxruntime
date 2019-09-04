import tensorflow as tf
from onnx import numpy_helper

image_size = 299
input_shape = [None, image_size, image_size, 3]

image_reader = tf.placeholder(tf.float32, shape=[None, None, 3])

def write_tensor(f,tensor,input_name=None):
    if input_name:
        tensor.name = input_name
    body = tensor.SerializeToString()
    f.write(body)

def read_tensor_from_image_file(image_path):
    file_reader = tf.read_file(image_path, "file_reader")
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    image_reader = tf.image.convert_image_dtype(image_reader, dtype=tf.float32)
    with tf.Session() as sess2:
        return sess2.run(image_reader)

#image_path = '/home/chasun/src/imagnet_validation_data/ILSVRC2012_val_00015060.JPEG'
image_path = '/home/chasun/src/imagnet_validation_data/ILSVRC2012_val_00018030.JPEG'

# Resize the image to the specified height and width.
image = tf.expand_dims(image_reader, 0)
image = tf.image.resize_bilinear(image, [image_size, image_size],
                                 align_corners=False)

with tf.Session() as sess:
    a = read_tensor_from_image_file(image_path)
    graph_output = sess.run(image,{
        image_reader: a
    })
    print(graph_output.shape)
    print(a.dtype)
    with open('/tmp/t/d1/b/input_0.pb', "wb") as f:
        t = numpy_helper.from_array(a)
        write_tensor(f,t)
    with open('/tmp/t/d1/b/output_0.pb', "wb") as f:
        t = numpy_helper.from_array(graph_output)
        write_tensor(f,t)
    tf.io.write_graph(sess.graph_def, '/tmp/a', 'test_graph.pb',as_text=False)