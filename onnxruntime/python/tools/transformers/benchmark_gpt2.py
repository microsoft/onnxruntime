# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script benchmarks gpt2 model with past state.
# For gpt2 model without past state, use benchmark.py to measure performance.

import os
import sys
import numpy
import time
import psutil
import argparse
import logging
import torch
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig

logger = logging.getLogger('')

# Map alias to a tuple of Model Class and pretrained model name
MODEL_CLASSES = {
    "gpt2": (GPT2Model, GPT2Tokenizer, "gpt2"),
    "distilgpt2": (GPT2LMHeadModel, GPT2Tokenizer, "distilgpt2"),
}

def dump_environment():
    if "OMP_NUM_THREADS" in os.environ:
        logger.info("OMP_NUM_THREADS={}".format(os.environ["OMP_NUM_THREADS"]))
    else:
        logger.info("no environment variable of OMP_NUM_THREADS")

    if "OMP_WAIT_POLICY" in os.environ:
        logger.info("OMP_WAIT_POLICY={}".format(os.environ["OMP_WAIT_POLICY"]))
    else:
        logger.info("no environment variable of OMP_WAIT_POLICY")


def setup_environment(use_openmp=False):
    # ATTENTION: these environment variables must be set before importing onnxruntime.
    if use_openmp:
        os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
    else:
        os.environ["OMP_NUM_THREADS"] = '1'

    os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'
    dump_environment()


def pytorch_inference(model, input_ids, past=None, total_runs=100):
    latency = []
    with torch.no_grad():
        for _ in range(total_runs):
            start = time.time()
            outputs = model(input_ids=input_ids, past=past)
            latency.append(time.time() - start)

    average_latency = sum(latency) * 1000 / len(latency)
    logger.info("PyTorch Inference time = {} ms".format(format(average_latency, '.2f')))
    return outputs, average_latency


def onnxruntime_inference(ort_session, input_ids, past=None, total_runs=100):
    ort_inputs = {'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy())}

    # TODO: pass input tensor stored in GPU
    if past is not None:
        for i, past_i in enumerate(past):
            ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past[i].cpu().numpy())

    latency = []
    for _ in range(total_runs):
        start = time.time()
        ort_outputs = ort_session.run(None, ort_inputs)
        latency.append(time.time() - start)

    average_latency = sum(latency) * 1000 / len(latency)
    logger.info("OnnxRuntime Inference time = {} ms".format(format(average_latency, '.2f')))

    return ort_outputs, average_latency


def inference(model, ort_session, input_ids, past=None, total_runs=100, verify_outputs=True):
    outputs, torch_latency = pytorch_inference(model, input_ids, past, total_runs)
    ort_outputs, ort_latency = onnxruntime_inference(ort_session, input_ids, past, total_runs)
    if verify_outputs:
        # verify results
        is_close = numpy.allclose(ort_outputs[0], outputs[0].cpu(), rtol=1e-05, atol=1e-04)
        logger.info(f'PyTorch and OnnxRuntime output 0 (last_state) are close: {is_close}')

        for layer in range(model.config.n_layer):
            is_close = numpy.allclose(ort_outputs[1 + layer], outputs[1][layer].cpu(), rtol=1e-05, atol=1e-04)
            logger.info(f'PyTorch and OnnxRuntime layer {layer} state (present_{layer}) are close:{is_close}')
    return torch_latency, ort_latency

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type',
                        required=True,
                        type=str,
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("-c",
                        "--cache_dir",
                        required=False,
                        type=str,
                        default="./cache_models",
                        help="Directory to cache pre-trained models")

    parser.add_argument("--onnx_dir",
                        required=False,
                        type=str,
                        default="./onnx_models",
                        help="Directory to store onnx models")

    parser.add_argument('--total_runs', required=False, type=int, help="total runs", default=100)

    parser.add_argument('--optimizer', required=False, action='store_true')
    parser.set_defaults(optimizer=False)

    parser.add_argument('--use_openmp', required=False, action='store_true')
    parser.set_defaults(use_openmp=False)

    parser.add_argument('--use_gpu', required=False, action='store_true')
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args


def setup_logger(verbose=True):
    # output logging to stdout
    log_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        log_handler.setFormatter(logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s'))
        logging_level = logging.DEBUG
    else:
        log_handler.setFormatter(logging.Formatter('%(filename)20s: %(message)s'))
        logging_level = logging.INFO
    log_handler.setLevel(logging_level)

    # Avoid duplicated handlers when runing this script in multiple cells of Jupyter Notebook.
    if not logger.hasHandlers():
        logger.addHandler(log_handler)

    logger.setLevel(logging_level)


def main():
    args = parse_arguments()
    setup_logger(args.verbose)
    dump_environment()

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    output_dir = args.onnx_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    use_torchscript = False
    (model_class, tokenizer_class, model_name) = MODEL_CLASSES[args.model_type]
    config = AutoConfig.from_pretrained(model_name, torchscript=use_torchscript, cache_dir=cache_dir)
    model = model_class.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=cache_dir)

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.to(device)

    inputs = tokenizer.encode_plus("Here is an example input for GPT2 model",
                                   add_special_tokens=True,
                                   return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    print("input_ids", input_ids)
    outputs = model(input_ids=input_ids, past=None)
    assert len(outputs) == 2
    print("output 0 shape", outputs[0].shape)
    print("past state shape", outputs[1][0].shape)

    num_layer = model.config.n_layer
    present_names = [f'present_{i}' for i in range(num_layer)]
    output_names = ["last_state"] + present_names

    input_names = ['input_ids']
    # input_ids has only one word for model with past state.
    dynamic_axes = {'input_ids': {0: 'batch_size'}, 'last_state': {0: 'batch_size', 1: 'seq_len'}}

    for name in present_names:
        dynamic_axes[name] = {1: 'batch_size', 3: 'seq_len'}

    past_names = [f'past_{i}' for i in range(num_layer)]
    input_names = ['input_ids'] + past_names
    dummy_past = [torch.zeros(list(outputs[1][0].shape), dtype=torch.float32, device=device) for _ in range(num_layer)]
    for name in past_names:
        dynamic_axes[name] = {1: 'batch_size', 3: 'seq_len'}
    print(f"vocab_size:{model.config.vocab_size}")

    dummy_input_ids = torch.randint(low=0, high=model.config.vocab_size - 1, size=(1, 1), dtype=torch.int64, device=device)
    print("dummy_input_ids", dummy_input_ids)
    export_inputs = (dummy_input_ids, tuple(dummy_past))


    export_model_path = os.path.join(output_dir, 'gpt2_past.onnx')

    # Let's run performance test on PyTorch before updating environment variable.
    input_ids = dummy_input_ids
    past = dummy_past

    #if use_torchscript:
    #    model = torch.jit.trace(model, (input_ids, past))
    outputs = pytorch_inference(model, input_ids, past, total_runs=args.total_runs)

    torch.onnx.export(model,
                      args=export_inputs,
                      f=export_model_path,
                      input_names=input_names,
                      output_names=output_names,
                      example_outputs=outputs,
                      dynamic_axes=dynamic_axes,
                      opset_version=11,
                      do_constant_folding=True,
                      verbose=False)

    # setup environment variables before importing onnxruntime.
    setup_environment(args.use_openmp)
    import onnxruntime

    onnx_model_path = export_model_path

    if args.optimizer:
        from optimizer import optimize_model
        m = optimize_model(onnx_model_path,
                           model_type='gpt2',
                           num_heads=config.num_attention_heads,
                           hidden_size=config.hidden_size,
                           opt_level=0,
                           optimization_options=None,
                           use_gpu=args.use_gpu)
        onnx_model_path = os.path.join(output_dir, 'gpt2_past_optimized.onnx')
        m.save_model_to_file(onnx_model_path)

    if args.use_gpu and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        logger.warning(
            "Please install onnxruntime-gpu package to test GPU inference.")

    sess_options = onnxruntime.SessionOptions()

    if args.use_openmp:
        sess_options.intra_op_num_threads = 1
    else:
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    logger.info(f"session option: intra_op_num_threads={sess_options.intra_op_num_threads}")

    logger.info(f"Start inferencing onnx model: {onnx_model_path}")
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options)

    for sequence_length in (8, 16, 32, 64, 128, 256, 512):
        past_shape = [2, 1, config.num_attention_heads, sequence_length, int(config.hidden_size / config.num_attention_heads)]
        dummy_past = [torch.rand(past_shape, dtype=torch.float32, device=device) for _ in range(num_layer)]
        torch_latency, ort_latency = inference(model, session, input_ids, dummy_past)
        logger.info(f"sequence_length={sequence_length}, torch_latency={torch_latency}, ort_latency={ort_latency}")

if __name__ == '__main__':
    main()
