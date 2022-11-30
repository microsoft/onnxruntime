import onnx
import onnxruntime as ort
import numpy as np
import os
import pickle
import argparse
import time
import psutil


def setup_session_option():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4 # 0 for verbose
    so.enable_profiling = False

    return so

def create_input(args):
    data_shape = (args.batch, args.seq_len)

    x = np.random.randint(low=0, high=10000, size=data_shape, dtype=np.int64)
    return {'input_ids': x}

def main(args):
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    print('rank: ', local_rank)

    if args.model_file is not None:
        model_file = args.model_file
    elif args.shard_prefix is not None:
        model_file = f'{args.shard_prefix}_{local_rank}.onnx'

    inputs = create_input(args)

    so = setup_session_option() 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('ROCMExecutionProvider',{'device_id':local_rank})])

    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        y = sess.run(None, inputs)

        if local_rank == 0 and i % interval == 0:
            cost_time = time.time() - end
            print(f'iters: {i} cost: {cost_time} avg: {cost_time/interval}')
            end = time.time()

    return


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--shard-prefix', type=str)
    parser.add_argument('--loop-cnt', type=int, default=1000)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--batch', type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
