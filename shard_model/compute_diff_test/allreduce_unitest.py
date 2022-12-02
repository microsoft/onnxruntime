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

INPUT_DATA_FILE = 'input-data-{}.pkl'
ORIGIN_MODEL_FILE = 'origin-model-{}.onnx'

def numpy_type_to_onnx_type(dtype):
    return NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]

def numpy_type_to_str(dtype):
    return str(dtype)

def generate_allreduce_model(dtype_str, world_size, data_shape):
    # generate input data
    for i in range(world_size):
        data = np.random.rand(*data_shape).astype(dtype_str)
        with open(INPUT_DATA_FILE.format(i), 'wb') as fp:
            pickle.dump(data, fp)

    onnx_type = numpy_type_to_onnx_type(dtype_str)
    # genrate single model
    A = helper.make_tensor_value_info('A', onnx_type, data_shape)
    Y = helper.make_tensor_value_info('Y', onnx_type, data_shape)

    coll_node = coll('NcclAllReduce', 'A', 'Y', group_type=0)
    origin_model = generate_model([coll_node], [A], [Y], None, 'origin-allreduce')

    onnx.checker.check_model(origin_model)

    origin_model_file = ORIGIN_MODEL_FILE.format(dtype_str)
    onnx.save(origin_model, origin_model_file)

    return origin_model_file


def run_model(input_data, model_file, device_id):
    so = ort.SessionOptions()
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('CUDAExecutionProvider',{'device_id': device_id})])

    y = sess.run(None, input_data)
    return y


def main(args):
    data_shape = (512, 768)
    if args.generate_model:
        origin_model_file=generate_allreduce_model(args.type, args.size, data_shape)
        print('origin model file: ', origin_model_file)
        return

    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    print('rank: ', local_rank)

    # load input data for each rank
    input_datas = []
    for i in range(args.size):
        with open(INPUT_DATA_FILE.format(i), 'rb') as fp:
            input_datas.append(pickle.load(fp))

    y = run_model({'A': input_datas[local_rank]}, ORIGIN_MODEL_FILE.format(args.type), local_rank)

    need_y = input_datas[0]
    for i in input_datas[1:]:
        need_y = need_y.astype(np.float32) + i.astype(np.float32)

    need_y = need_y.astype(args.type)

    if np.allclose(need_y, y[0]):
        print(f'SAME. shape: {y[0].shape}')
    else:
        diff = abs(need_y - y[0])
        rel_diff = abs(diff / y[0])
        print(f'NOT SAME. shape: {y[0].shape} diff: {diff.max()} rel-diff: {rel_diff.max()}')


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--generate-model', action='store_true', default=False)
    parser.add_argument('--type', type=str, default='float16')
    parser.add_argument('--size', type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)

