#!/usr/bin/python3

import subprocess
import sys
import os
import shutil

TENSORFLOW_ROOT='/home/chasun/os/tensorflow'

input_name='input'
output_name='InceptionV3/Predictions/Reshape_1'
model_name='inception_v3'

output_dir = model_name
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)

subprocess.run([sys.executable, '/home/chasun/os/models/research/slim/export_inference_graph.py', '--alsologtostderr', '--model_name='+model_name,
                '--output_file=/data/tmp/graph_only.pb.pb'], check=True)

subprocess.run([os.path.join(TENSORFLOW_ROOT,'bazel-bin/tensorflow/python/tools/freeze_graph'), '--input_graph=/data/tmp/graph_only.pb.pb',
                '--input_checkpoint=/data/tf_checkpoints/inception_v3.ckpt',
                '--input_binary=true',
                '--output_graph=/data/tmp/frozen_graph.pb',
                '--output_node_names='+output_name],
               check=True)

subprocess.run([sys.executable, '-m', 'tf2onnx.convert', '--input', '/data/tmp/frozen_graph.pb', '--inputs', input_name + ':0',
                '--outputs', output_name+':0',  '--opset=8', '--fold_const', '--verbose', '--output', os.path.join(output_dir,'model.onnx')], check=True)
