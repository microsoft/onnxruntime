import onnx
import onnxruntime as ort
from onnxruntime import OrtValue
import numpy as np
import os
import pickle
import argparse
import time
import psutil


def setup_session_option(args):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4 # 0 for verbose
    so.enable_profiling = args.profile

    return so

def use_msccl():
    import msccl
    #msccl.init('ndv2', 1, (msccl.Collective.alltoall, ('1MB')))
    msccl.init('ndv4', 1, (msccl.Collective.allreduce, '1MB'))

def create_input(args):
    #data_shape = (args.batch, args.seq_len)

    #x = np.random.randint(low=0, high=10000, size=data_shape, dtype=np.int64)
    #return {'input_ids': x}
    mpi = False
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        mpi = True

    input_file = 'inputs.pkl'
    if mpi:
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 1))
        input_file = f'inputs-{local_rank}.pkl'

    with open(input_file, 'rb') as fp:
        inputs = pickle.load(fp)
    return inputs

def main(args):
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 1))
    print('rank: ', local_rank)

    if args.use_msccl:
        use_msccl()

    if args.model_file is not None:
        model_file = args.model_file
    elif args.shard_prefix is not None:
        model_file = f'{args.shard_prefix}_{local_rank}.onnx'

    inputs = create_input(args)

    so = setup_session_option(args) 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('CUDAExecutionProvider',{'device_id':local_rank})])
    io_binding = sess.io_binding()

    # bind inputs by using OrtValue
    for k in inputs:
        x = OrtValue.ortvalue_from_numpy(inputs[k], 'cuda', local_rank)
        io_binding.bind_ortvalue_input(k, x)
    # bind outputs
    outputs = sess.get_outputs()
    for out in outputs:
        io_binding.bind_output(out.name, 'cuda', local_rank)


    end = time.time()
    interval = args.interval
    for i in range(args.loop_cnt):
        #y = sess.run(None, inputs)
        sess.run_with_iobinding(io_binding)

        if local_rank == 1 and i % interval == 0:
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
    parser.add_argument('--use-msccl', action='store_true', default=False)
    parser.add_argument('--profile', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
