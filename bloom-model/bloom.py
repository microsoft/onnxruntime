import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

import os
import psutil
import time
import argparse
import onnx
import numpy as np

import onnxruntime as ort
from onnxruntime import OrtValue
from onnxruntime.transformers.optimizer import optimize_by_fusion
from onnxruntime.transformers.fusion_options import FusionOptions

from transformers import BloomConfig, BloomTokenizerFast
from models.modeling_bloom import BloomModel


def init_dist(args):
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK',0))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:9876', world_size=world_size, rank=rank)
    device = torch.device(local_rank)
    return device

def get_rank():
    rank = 0
    if 'LOCAL_RANK' in os.environ:
        rank = int(os.environ.get('LOCAL_RANK', '0'))
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    return rank

def get_process_group():
    return _get_default_group()



def setup_session_option(args, local_rank):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4 # 0 for verbose
    so.enable_profiling = args.profile
    so.profile_file_prefix=f'ort-profile-rank{local_rank}'

    return so

def run_onnxruntime(args, model_file, inputs):
    local_rank = get_rank()
    print('infer ort in rank: ', local_rank, ' m: ', model_file)
    so = setup_session_option(args, local_rank) 
    sess = ort.InferenceSession(model_file, sess_options=so, providers=[('ROCMExecutionProvider',{'device_id':local_rank})])
    io_binding = sess.io_binding()

    # bind inputs by using OrtValue
    input_names = sess.get_inputs()
    for k in input_names:
        np_data = inputs[k.name].cpu().numpy()
        x = OrtValue.ortvalue_from_numpy(np_data, 'cuda', local_rank)
        io_binding.bind_ortvalue_input(k.name, x)
    # bind outputs
    outputs = sess.get_outputs()
    for out in outputs:
        io_binding.bind_output(out.name, 'cuda', local_rank)

    sess.run_with_iobinding(io_binding)

    output = io_binding.copy_outputs_to_cpu()
    return output

    #end = time.time()
    #interval = args.interval
    #for i in range(args.loop_cnt):
    #    #y = sess.run(None, inputs)
    #    sess.run_with_iobinding(io_binding)

    #    if local_rank == 1 and i % interval == 0:
    #        cost_time = time.time() - end
    #        print(f'iters: {i} cost: {cost_time} avg: {cost_time/interval}')
    #        end = time.time()

    #return

def get_dummy_inputs(batch, seq_len, past_seq_len, config, device):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch, seq_len), dtype=torch.int64, device=device)
    att_mask = torch.ones((batch, seq_len), dtype=torch.int64, device=device)
    inputs = (input_ids, att_mask)
    input_names = ['input_ids', 'attention_mask']
    return {k: v for k, v in zip(input_names, inputs)}, input_names
   
def get_bloom_model(name):
    config = BloomConfig.from_pretrained(name)
    config.n_layer = 1
    process_group=get_process_group()
    model = BloomModel(config, process_group=process_group)
    return config, model


def main(args):
    device=init_dist(args)
    batch=1
    seq_len=128
    local_rank = get_rank()

    torch.cuda.set_device(device)
    #torch.cuda.manual_seed(42)

    config, model = get_bloom_model('bigscience/bloom-560m')

    model.to(device)
    model.requires_grad_(False)

    inputs, input_names = get_dummy_inputs(batch, seq_len, None, config, device)
    output_names = ['output']

    # try forward
    with torch.no_grad():
        output = model(**inputs)
        output = output.last_hidden_state
    print('output: ', output.shape, ' dtype:', output.dtype, ' dev:', output.device)

    model_out_file = f'rank-{local_rank}-{args.output}'

    # export to onnx
    tmp_file = f'tmp-{model_out_file}'
    torch.onnx.enable_log()
    torch.onnx.export(
            model,
            f=tmp_file,
            args=inputs,
            input_names=input_names,
            output_names=output_names,
            opset_version=15,
            verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            custom_opsets={'com.microsoft':1},
            export_params=True, # need these config to keep result same as torch
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(tmp_file)
    onnx.save(onnx_model, model_out_file, save_as_external_data=True, location=f'{model_out_file}.data')

    print('export to onnx done.')

    ## use fusion to optimize model
    model_type = 'gpt2'
    opt_option=FusionOptions(model_type)
    optimizer = optimize_by_fusion(
            onnx.load(model_out_file), 
            model_type=model_type,
            num_heads=config.n_head,
            hidden_size=config.hidden_size,
            optimization_options=opt_option
        )

    opt_out_file = f'opt-{model_out_file}'
    optimizer.save_model_to_file(opt_out_file, use_external_data_format=True)

    print('save optimized onnx done.')

    #ort_out = run_onnxruntime(args, args.output, {k: v for k, v in zip(input_names, inputs)})

    #o1 = output.cpu().numpy()

    #if np.allclose(o1, ort_out[0]):
    #    print('result SAME.')
    #else:
    #    diff = abs(o1 - ort_out[0])
    #    rel_diff = abs(diff / o1)
    #    print(f'not SAME, max diff: {diff.max()}, rel-diff: {rel_diff.max()}')


def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--input', type=str, help='input a file that contains profile result file names')
    parser.add_argument('--output', type=str, help='output file name')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--loop-cnt', type=int, default=1000)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--profile', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arges()
    main(args)
