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

input_name = 'input_tensor'
output_names = ['ArgMax',
                'softmax_tensor']
input_height = 224
input_width = 224
print('load graph into tensorflow')
#image_folder = '/home/chasun/Downloads/imagenet'
image_folder = '/home/chasun/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min'
config = tf.ConfigProto()
image_count = 0
prog = re.compile('ILSVRC2012_val_(\\d+).JPEG')

def write_tensor(f,tensor,input_name=None):
    if input_name:
        tensor.name = input_name
    body = tensor.SerializeToString()
    f.write(body)

output_dir = "."

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

        image_from_tf = read_tensor_from_image_file(
            preprocessing_fn,
            image_path,
            input_height=input_height,
            input_width=input_width)
        t = np.expand_dims(image_from_tf, axis=0)
        print(t.shape)
          test_data_set_dir = os.path.join(output_dir,'test_data_set_%s' % data_set_id)
        os.makedirs(test_data_set_dir)

        im_np = np.expand_dims(np.transpose(image_from_tf, (2, 0, 1)), axis=0)
        #now im_np is in NCHW format
        with  open(os.path.join(test_data_set_dir, "input_0.pb"), "wb") as f:
            t = numpy_helper.from_array(im_np.astype(np.float32))
            write_tensor(f,t,input_name+":0")

        if image_count >= 100:
            break
