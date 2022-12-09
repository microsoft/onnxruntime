import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np
import os
import pickle
import argparse
import time
import psutil

def modify_model_outputs(model_file, modified_file, output_names):
    ori_model = onnx.load(model_file)
    value_info = {}
    for v in ori_model.graph.value_info:
        value_info[v.name]=v

    for n in output_names:
        if n in value_info:
            ori_model.graph.output.append(value_info[n])
        else:
            v = helper.make_tensor_value_info(n, TensorProto.FLOAT16, [None])
            ori_model.graph.output.append(v)

    onnx.save(ori_model, modified_file)

def run_model(model_file, inputs, device_id=None, output_names=None):
    so = ort.SessionOptions()
    if device_id is not None:
        sess = ort.InferenceSession(model_file, sess_options=so, providers=[('CUDAExecutionProvider',{'device_id':device_id})])
    else:
        sess = ort.InferenceSession(model_file, sess_options=so, providers=['CPUExecutionProvider'])

    res = sess.run(output_names, inputs)
    return res

def main(args):
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    print('rank: ', local_rank)

    origin_model_file = args.model_file
    model_file = f'{args.shard_prefix}_{local_rank}.onnx'
    input_file = 'inputs.pkl'
    with open(input_file, 'rb') as fp:
        x = pickle.load(fp)

    inputs = x

    output_names = ['EmbedLayerNormalization_0_output', 'onnx::MatMul_336', 'onnx::Add_338']
    save_mid_output = False

    if save_mid_output and local_rank == 0:
        modified_model_file = 'origin_bert-add-out.onnx'
        modify_model_outputs(origin_model_file, modified_model_file, output_names)
        origin_res = run_model(modified_model_file, inputs, local_rank, output_names)
        
        with open('ori-added-out-res.pkl', 'wb') as fp:
            pickle.dump(origin_res, fp)
    else:
        origin_res = run_model(origin_model_file, inputs, local_rank)

    # load and run sharded model
    if save_mid_output:
        modified_model_file = f'bert-rank-{local_rank}-add-out.onnx'
        modify_model_outputs(model_file, modified_model_file, output_names)
        res = run_model(modified_model_file, inputs, local_rank, output_names)
        with open(f'rank-{local_rank}-res.pkl', 'wb') as fp:
            pickle.dump(res, fp)
        return

    with open(f'inputs-{local_rank}.pkl', 'rb') as fp:
        inputs = pickle.load(fp)

    res = run_model(model_file, inputs, local_rank)

    for r0, r1 in zip(origin_res, res):
        if np.allclose(r0, r1):
            print(f'r0 {r0.shape} r1 {r1.shape} is same.')
        else:
            diff = r0 - r1
            rel_diff = diff / r1
            print(f'r0 {r0.shape} r1 {r1.shape} is not same. max: {diff.max()}, rel-diff: {rel_diff.max()}')
        break

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--shard-prefix', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
