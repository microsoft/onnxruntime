import os
import argparse
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto, OperatorSetIdProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
import onnxruntime as ort
import numpy as np
from generate_model import generate_model, coll, matmul
import pickle

INPUT_DATA_FILE = 'input-data.pkl'
RANK_DATA_FILE = 'input-data-{}.pkl'
ORIGIN_MODEL_FILE = 'origin-model-{}.onnx'
RANK_MODEL_FILE = 'model-{}-rank-{}.onnx'

def numpy_type_to_onnx_type(dtype):
    return NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]

def numpy_type_to_str(dtype):
    return str(dtype)

def generate_k_split_model(dtype_str, m, n, k, ranks):
    assert k % ranks == 0
    unit_size = int(k / ranks)

    # generate input data for each rank
    a = np.random.rand(m, k).astype(dtype_str)
    with open(INPUT_DATA_FILE, 'wb') as fp:
        pickle.dump(a, fp)

    # split a by k
    for r in range(ranks):
        slc = [slice(None), slice(unit_size * r,  unit_size * r + unit_size)]
        a_split = a[tuple(slc)]
        with open(RANK_DATA_FILE.format(r), 'wb') as fp:
            pickle.dump(a_split, fp)

    # generate random tensro for B[k,n]
    B = np.random.rand(k, n).astype(dtype_str)

    B_onnx = onnx.numpy_helper.from_array(B, name='B')

    onnx_type = numpy_type_to_onnx_type(dtype_str)
    # genrate single model
    A = helper.make_tensor_value_info('A', onnx_type, [m, k])
    Y = helper.make_tensor_value_info('Y', onnx_type, [m, n])

    node = matmul('A', 'B', 'Y')
    origin_model = generate_model([node], [A], [Y], [B_onnx], 'origin-matmul')

    onnx.checker.check_model(origin_model)

    origin_model_file = ORIGIN_MODEL_FILE.format(dtype_str)
    onnx.save(origin_model, origin_model_file)

    B_rank = np.split(B, ranks, axis=0)
    # generate split model
    # split B's k dim
    rank_model_files = []
    for r in range(ranks):
        # generate initializer data
        B_onnx = onnx.numpy_helper.from_array(B_rank[r], name='B')

        # generate matmul node
        A = helper.make_tensor_value_info('A', onnx_type, [m, unit_size])
        Y = helper.make_tensor_value_info('Y', onnx_type, [m, n])

        node = matmul('A', 'B', 'Y')

        rank_model = generate_model([node], [A], [Y], [B_onnx], f'rank-{r}-matmul')

        onnx.checker.check_model(rank_model)
        rank_m_file = RANK_MODEL_FILE.format(dtype_str, r)
        onnx.save(rank_model, rank_m_file)
        rank_model_files.append(rank_m_file)

    return origin_model_file, rank_model_files


def run_model(input_data, model_file, device_id):
    so = ort.SessionOptions()
    #so.log_severity_level = 0  # verbose
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('CUDAExecutionProvider',{'device_id': device_id})])
    #sess = ort.InferenceSession(model_file, sess_options=so, providers=[('CPUExecutionProvider')])

    y = sess.run(None, input_data)
    return y

def single_run(args):
    # load and run rank model
    res = []
    for rank in range(args.size):
        with open(RANK_DATA_FILE.format(rank), 'rb') as fp:
            rank_data = pickle.load(fp)

        rank_y = run_model({'A': rank_data}, RANK_MODEL_FILE.format(args.type, rank), 0)
        res.append(rank_y)

    # do sum
    for r in res[1:]:
        for i, r1 in enumerate(r):
            res[0][i] = res[0][i].astype(np.float32) + r1.astype(np.float32)

    return [r.astype(args.type) for r in res[0]]

def main(args):
    origin_model_file, rank_files=generate_k_split_model(args.type, args.m, args.n, args.k, args.size)
    print('origin model file: ', origin_model_file)
    assert len(rank_files) == args.size
    print('rank model files: ')
    for f in rank_files:
        print(f)

    # load and run origin model
    with open(INPUT_DATA_FILE, 'rb') as fp:
        data = pickle.load(fp)

    origin_y = run_model({'A': data}, ORIGIN_MODEL_FILE.format(args.type), 0)

    # load and run rank model
    rank_y = single_run(args)

    if np.allclose(origin_y[0], rank_y[0]):
        print(f'SAME. orign shape: {origin_y[0].shape}, rank shape: {rank_y[0].shape}')
    else:
        diff = abs(origin_y[0] - rank_y[0])
        rel_diff = abs(diff / rank_y[0])
        print(f'NOT SAME. orign shape: {origin_y[0].shape}, rank shape: {rank_y[0].shape}, diff: {diff.max()}, rel_diff: {rel_diff.max()}')


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--type', type=str, default='float16')
    parser.add_argument('--m', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--size', type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)

