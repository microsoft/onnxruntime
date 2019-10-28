#!/usr/bin/python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from preprocessing import preprocessing_factory
import argparse
import time
import numpy as np
import subprocess
import sys
import os
import re
import shutil
import tensorflow as tf
import onnx
from onnx import utils
from onnx import numpy_helper


def load_tensorflow_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(preprocessing_fn, file_name,
                                input_height,
                                input_width):
  print("input_height=%d" % input_height)
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
  normalized = preprocessing_fn(image_reader, input_height,input_width)

  with tf.Session() as sess:
    return sess.run(normalized)


tf_checkpoint_dir = '/data/testdata/tf_checkpoints'

if __name__ == "__main__":
  input_height = 299
  input_width = 299

  parser = argparse.ArgumentParser()

  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--opset", type=int, nargs='+', help="7,8")
  parser.add_argument("--input_layer", type=str)
  parser.add_argument("--output_layer", type=str)
  parser.add_argument("--model_name", type=str)
  args = parser.parse_args()


  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width


  TENSORFLOW_ROOT='/home/chasun/os/tensorflow'

  input_name=args.input_layer
  output_name=args.output_layer
  model_name=args.model_name
  model_file = os.path.join(tf_checkpoint_dir, model_name+"_frozen.pb")
  if os.path.exists(model_file):
     print("skip freeze step")
     shutil.copyfile(model_file, '/data/tmp/frozen_graph.pb')
  else:
    graph_only_model_file = os.path.join(tf_checkpoint_dir, model_name+".pbtxt")
    has_text_model_file=False
    if os.path.exists(graph_only_model_file):
       has_text_model_file=True
    else:
      graph_only_model_file='/data/tmp/graph_only.pb.pb'
      export_args = [sys.executable, '/home/chasun/os/models/research/slim/export_inference_graph.py', '--image_size=%s' % input_height,'--alsologtostderr', '--model_name='+model_name,
                 '--output_file='+graph_only_model_file]
      if model_name.startswith('resnet_v1') or model_name.startswith('vgg'):
        export_args.append('--labels_offset=1')
      subprocess.run(export_args, check=True)
    model_file = '/data/tmp/frozen_graph.pb'
    freeze_args = ['freeze_graph', '--input_graph='+graph_only_model_file,
                  '--input_checkpoint=%s.ckpt' % os.path.join(tf_checkpoint_dir, model_name),    
                  '--output_graph=' + model_file,
                  '--output_node_names='+output_name]
    if has_text_model_file:
       freeze_args.append('--input_binary=false')
    else:
       freeze_args.append('--input_binary=true')
    subprocess.run(freeze_args, check=True)
  for opset in args.opset:
    print("opset: %d" % opset)
    output_dir = os.path.join("opset%d" % opset, model_name)
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    shutil.copyfile('/data/tmp/frozen_graph.pb',os.path.join(output_dir,'model.tf.pb'))
    with open(os.path.join(output_dir,'model.tf.pb.meta'), "w") as text_file:
      text_file.write("input=%s:0\n" % input_name)
      text_file.write("output=%s:0\n" % output_name)

    model_without_raw_data = os.path.join(output_dir,'model.onnx')
    tf2onnx_cmd = [sys.executable, '-m', 'tf2onnx.convert', '--input', model_file, '--inputs', input_name + ':0',
                    '--outputs', output_name+':0',  '--opset=%d' % opset, '--verbose', '--output', model_without_raw_data]
    print("running: %s"  % tf2onnx_cmd)
    subprocess.run(tf2onnx_cmd, check=True)
    #print('optimize model')
    #onnx_model = onnx.load(model_without_raw_data)
    #model_without_raw_data = onnx.utils.polish_model(onnx_model)
    #onnx_model = onnx.utils.polish_model(onnx_model)
    #onnx.external_data_helper.convert_model_to_external_data(onnx_model, True, 'raw_data')
    #onnx.save(onnx_model, model_without_raw_data)

    print('load graph into tensorflow')
    graph = load_tensorflow_graph(model_file)
    image_folder = '/home/chasun/src/imagnet_validation_data'
    input_operation = graph.get_operation_by_name("import/" + input_name)
    output_operation = graph.get_operation_by_name("import/" + output_name)
    config = tf.ConfigProto()
    image_count = 0
    prog = re.compile('ILSVRC2012_val_(\\d+).JPEG')
    with tf.Session(graph=graph,config=config) as sess:
      for image_file_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file_name)
        if not os.path.isfile(image_path):
          continue
        re_result = prog.match(image_file_name)
        if re_result is None:
          print('skip %s' % image_path)
          continue
        image_count +=1
        data_set_id = re_result[1]
        if model_name.startswith('resnet_v2') or model_name.startswith('mobilenet'):
          preprocessing_fn = preprocessing_factory.get_preprocessing('inception')
        else:
          preprocessing_fn = preprocessing_factory.get_preprocessing(model_name)

        t = read_tensor_from_image_file(
            preprocessing_fn,
            image_path,
            input_height=input_height,
            input_width=input_width)
        t = np.expand_dims(t, axis=0)
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
        test_data_set_dir = os.path.join(output_dir,'test_data_set_%s' % data_set_id)
        os.makedirs(test_data_set_dir)
        im = numpy_helper.from_array(t)
        im.name = input_name + ":0"
        with  open(os.path.join(test_data_set_dir,"input_0.pb"), "wb") as f:
          f.write(im.SerializeToString())

        result_tensor = numpy_helper.from_array(results)
        with  open(os.path.join(test_data_set_dir,"output_0.pb"), "wb") as f:
         f.write(result_tensor.SerializeToString())
        if image_count >= 10:
          break
  print('finished')



