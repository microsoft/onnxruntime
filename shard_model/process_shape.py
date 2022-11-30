import onnx
from onnx import helper, shape_inference
import argparse
import pickle
import numpy as np

def process_shape(batch, seq_len, model):
    for input in model.graph.input:
        new_shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param == "batch_size":
                new_shape.append(batch)
            elif dim.dim_param == "seq_len":
                new_shape.append(seq_len)
            else:
                new_shape.append(dim.dim_value)
        del input.type.tensor_type.shape.dim[:]
        for shape in new_shape:
            dim = input.type.tensor_type.shape.dim.add()
            dim.dim_value = shape

    for output in model.graph.output:
        new_shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_param == "batch_size":
                new_shape.append(batch)
            elif dim.dim_param == "seq_len":
                new_shape.append(seq_len)
            else:
                new_shape.append(dim.dim_value)
        del output.type.tensor_type.shape.dim[:]
        for shape in new_shape:
            dim = output.type.tensor_type.shape.dim.add()
            dim.dim_value = shape

    for value_info in model.graph.value_info:
        new_shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_param == "batch_size":
                new_shape.append(batch)
            elif dim.dim_param == "seq_len":
                new_shape.append(seq_len)
            else:
                new_shape.append(dim.dim_value)
        del value_info.type.tensor_type.shape.dim[:]
        for shape in new_shape:
            dim = value_info.type.tensor_type.shape.dim.add()
            dim.dim_value = shape
    return model

def main(args):
    model = onnx.load(args.input)
    model = process_shape(args.batch, args.seq_len, model)

    onnx.save(model, args.output)

    # generate fake input data
    data_shape = (args.batch, args.seq_len)

    x = np.random.randint(low=0, high=10000, size=data_shape, dtype=np.int64)
    with open('input-x.pkl', 'wb') as fp:
        pickle.dump(x, fp)

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--seq-len', type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
