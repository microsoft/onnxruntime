import onnx
import os
from onnx import helper, shape_inference
import argparse
import pickle
import numpy as np
import torch
import random
from transformers import AutoConfig

def process_shape(model, batch, seq_len, past_seq_len):
    total_seq_len = seq_len + past_seq_len
    dim_map = {
            'batch_size': batch,
            'seq_len': seq_len,
            'total_seq_len': total_seq_len,
            'past_seq_len': past_seq_len
            }
    for input in model.graph.input:
        new_shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param in dim_map:
                new_shape.append(dim_map[dim.dim_param])
            else:
                new_shape.append(dim.dim_value)
        del input.type.tensor_type.shape.dim[:]
        for shape in new_shape:
            dim = input.type.tensor_type.shape.dim.add()
            dim.dim_value = shape

    for output in model.graph.output:
        new_shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_param in dim_map:
                new_shape.append(dim_map[dim.dim_param])
            else:
                new_shape.append(dim.dim_value)
        del output.type.tensor_type.shape.dim[:]
        for shape in new_shape:
            dim = output.type.tensor_type.shape.dim.add()
            dim.dim_value = shape

    for value_info in model.graph.value_info:
        new_shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_param in dim_map:
                new_shape.append(dim_map[dim.dim_param])
            else:
                new_shape.append(dim.dim_value)
        del value_info.type.tensor_type.shape.dim[:]
        for shape in new_shape:
            dim = value_info.type.tensor_type.shape.dim.add()
            dim.dim_value = shape
    return model

def main(args):
    model = onnx.load(args.input)
    model = process_shape(model, args.batch, args.seq_len, args.past_seq_len)

    output_file = os.path.split(args.output)[1]
    external_data_file = f'{output_file}.data'
    onnx.save(model,
            args.output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_file
    )


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--past-seq-len', type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
