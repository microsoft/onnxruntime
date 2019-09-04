import tensorflow as tf
import sys
from onnx import numpy_helper

file_name = sys.argv[1]
input_height=224
input_width=224
input_mean=0
input_std=255
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
  image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
float_caster = tf.cast(image_reader, tf.float32)
dims_expander = tf.expand_dims(float_caster, 0)
resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

im = None
with tf.Session() as sess:
    im = sess.run(normalized)
im = numpy_helper.from_array(im)
#im.name="input:0"
im.name="image"
with  open("input_0.pb", "wb") as f:
   f.write(im.SerializeToString())
