# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import argparse
import cntk as C
from model_editor_internal import PairDescription
import numpy as np
import onnx
from onnx import numpy_helper
import os

def save_data(test_data_dir, var_and_data, output_uid_to_ort_pairs=None):
    for i, (var, data) in enumerate(var_and_data.items()):
        data = np.asarray(data).astype(var.dtype)
        # ONNX input shape always has sequence axis as the first dimension, if sequence axis exists
        if len(var.dynamic_axes) == 2:
            data = data.transpose((1,0,)+tuple(range(2,len(data.shape))))
        file_path = os.path.join(test_data_dir, '{}_{}.pb'.format('output' if output_uid_to_ort_pairs else 'input', i))
        if output_uid_to_ort_pairs:
            tensor_name = output_uid_to_ort_pairs[var.uid]
        else:
            tensor_name = var.name if var.name else var.uid
        onnx.save_tensor(numpy_helper.from_array(data, tensor_name), file_path)


def convert_model_and_gen_data(input, output, end_node, seq_len, batch_size):
    cntk_model = C.load_model(input)
    if end_node:
        nodes = C.logging.depth_first_search(cntk_model, lambda x: x.name == end_node, depth=-1)
        assert len(nodes) == 1
        cntk_model = C.as_composite(nodes[0])
    cntk_model.save(output, C.ModelFormat.ONNX)

    if seq_len==0:
        return

    pair_desc = PairDescription()
    pair_string = onnx.load(output).graph.doc_string
    pair_desc.parse_from_string(pair_string)

    cntk_feeds = {}
    for var in cntk_model.arguments:
        data_shape = []
        for ax in var.dynamic_axes:
            if ax.name == 'defaultBatchAxis':
                data_shape = data_shape + [batch_size]
            else:
                data_shape = data_shape + [seq_len] # TODO: handle models with multiple sequence axes
        data_shape = data_shape + list(var.shape)
        cntk_feeds[var] = np.random.rand(*data_shape).astype(var.dtype)

    # run inference with CNTK
    cntk_output = cntk_model.eval(cntk_feeds)
    if type(cntk_output) != dict:
        assert len(cntk_model.outputs) == 1
        cntk_output = {cntk_model.output : cntk_output}

    test_data_dir = os.path.join(os.path.split(output)[0], 'test_data_set_0')
    os.makedirs(test_data_dir, exist_ok=True)
    save_data(test_data_dir, cntk_feeds)
    save_data(test_data_dir, cntk_output, pair_desc.get_pairs(PairDescription.PairType.uid_2_onnx_node_name))


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', help='The input CNTK model file.', required=True, default=None)
  parser.add_argument('--output', help='The output ONNX model file.', required=True, default=None)
  parser.add_argument('--end_node', help='The end node of CNTK model. This is to remove error/loss related parts from input model.', default=None)
  parser.add_argument('--seq_len', help='Test data sequence length.', type=int, default=0)
  parser.add_argument('--batch_size', help='Test data batch size.', type=int, default=1)
  return parser.parse_args()


if __name__ == '__main__':
    C.try_set_default_device(C.cpu())
    args = parse_arguments()
    print('input model: ' + args.input)
    print('output model: ' + args.output)
    convert_model_and_gen_data(args.input, args.output, args.end_node, args.seq_len, args.batch_size)
    print('Done!')
