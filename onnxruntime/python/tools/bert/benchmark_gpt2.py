# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import sys
import numpy
import time
import psutil
import argparse
import logging
import torch
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger('')

# Map alias to a tuple of Model Class and pretrained model name
MODEL_CLASSES = {
    "gpt2": (GPT2Model, GPT2Tokenizer, "gpt2"),
    "distilgpt2": (GPT2LMHeadModel, GPT2Tokenizer, "distilgpt2")
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

    logger.info("PyTorch Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))
    return outputs


def onnxruntime_inference(ort_session, input_ids, past=None, total_runs=100):
    # Use contiguous array as input might improve performance.
    # You can check the results from performance test tool to see whether you need it.
    ort_inputs = {'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy())}

    if past is not None:
        for i, past_i in enumerate(past):
            ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past[i].cpu().numpy())

    latency = []
    for _ in range(total_runs):
        start = time.time()
        ort_outputs = ort_session.run(None, ort_inputs)
        latency.append(time.time() - start)

    logger.info("OnnxRuntime Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

    return ort_outputs


def inference(model, ort_session, input_ids, past=None, total_runs=100, verify_outputs=True):
    outputs = pytorch_inference(model, input_ids, past, total_runs)
    ort_outputs = onnxruntime_inference(ort_session, input_ids, past, total_runs)
    if verify_outputs:
        logger.info('PyTorch and OnnxRuntime output 0 (last_state) are close:'.format(0),
                    numpy.allclose(ort_outputs[0], outputs[0].cpu(), rtol=1e-05, atol=1e-04))

        for layer in range(model.config.n_layer):
            logger.info('PyTorch and OnnxRuntime layer {} state (present_{}) are close:'.format(layer, layer),
                        numpy.allclose(ort_outputs[1 + layer], outputs[1][layer].cpu(), rtol=1e-05, atol=1e-04))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type',
                        required=True,
                        type=str,
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--cache_dir', required=True, type=str, help="cache directory")

    parser.add_argument('--output_dir', required=True, type=str, help="output onnx model directory")

    parser.add_argument('--total_runs', required=False, type=int, help="total runs", default=100)

    parser.add_argument('--enable_past_input', required=False, action='store_true')
    parser.set_defaults(enable_past_input=False)

    parser.add_argument('--enable_optimization', required=False, action='store_true')
    parser.set_defaults(enable_optimization=False)

    parser.add_argument('--verify_outputs', required=False, action='store_true')
    parser.set_defaults(verify_outputs=False)

    parser.add_argument('--use_openmp', required=False, action='store_true')
    parser.set_defaults(use_openmp=False)

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


def remove_past_outputs(export_model_path):
    from onnx import ModelProto
    from OnnxModel import OnnxModel

    model = ModelProto()
    with open(export_model_path, "rb") as f:
        model.ParseFromString(f.read())
    bert_model = OnnxModel(model)

    # remove past state outputs and only keep the first output.
    keep_output_names = [bert_model.model.graph.output[0].name]
    logger.info(f"Prune graph to keep the first output and drop past state outputs:{keep_output_names}")
    bert_model.prune_graph(keep_output_names)
    onnx_model_path = os.path.join(output_dir, 'gpt2_past{}_out1.onnx'.format(int(enable_past_input)))
    bert_model.save_model_to_file(onnx_model_path)
    return onnx_model_path


def main():
    args = parse_arguments()
    setup_logger(args.verbose)
    dump_environment()

    enable_past_input = args.enable_past_input

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    (model_class, tokenizer_class, model_name_or_path) = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model = model_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model.eval().cpu()

    inputs = tokenizer.encode_plus("Here is an example input for GPT2 model",
                                   add_special_tokens=True,
                                   return_tensors='pt')
    input_ids = inputs['input_ids']
    outputs = model(input_ids=input_ids, past=None)

    num_layer = model.config.n_layer
    present_names = [f'present_{i}' for i in range(num_layer)]
    output_names = ["last_state"] + present_names

    input_names = ['input_ids']
    dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'seq_len'}, 'last_state': {0: 'batch_size', 1: 'seq_len'}}
    for name in present_names:
        dynamic_axes[name] = {1: 'batch_size', 3: 'seq_len'}

    if enable_past_input:
        past_names = [f'past_{i}' for i in range(num_layer)]
        input_names = ['input_ids'] + past_names
        dummy_past = [torch.zeros(list(outputs[1][0].shape)) for _ in range(num_layer)]
        for name in past_names:
            dynamic_axes[name] = {1: 'batch_size', 3: 'seq_len'}
        export_inputs = (inputs['input_ids'], tuple(dummy_past))
    else:
        export_inputs = (inputs['input_ids'])

    export_model_path = os.path.join(output_dir, 'gpt2_past{}.onnx'.format(int(enable_past_input)))

    torch.onnx.export(model,
                      args=export_inputs,
                      f=export_model_path,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11,
                      do_constant_folding=True,
                      verbose=False)

    # Let's run performance test on PyTorch before updating environment variable.
    past = dummy_past if enable_past_input else None
    outputs = pytorch_inference(model, input_ids, past, total_runs=args.total_runs)

    # setup environment variables before importing onnxruntime.
    setup_environment(args.use_openmp)
    import onnxruntime

    onnx_model_path = export_model_path if enable_past_input else remove_past_outputs(export_model_path)

    if args.enable_optimization:
        from bert_model_optimization import optimize_model
        m = optimize_model(onnx_model_path,
                           model_type='gpt2',
                           gpu_only=False,
                           num_heads=12,
                           hidden_size=768,
                           sequence_length=64,
                           input_int32=False,
                           float16=False,
                           opt_level=0)
        onnx_model_path = os.path.join(output_dir, 'gpt2_past{}_optimized.onnx'.format(int(enable_past_input)))
        m.save_model_to_file(onnx_model_path)

    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        logger.warning(
            "onnxruntime-gpu is not built with OpenMP. You might try onnxruntime package to test CPU inference.")

    sess_options = onnxruntime.SessionOptions()

    if args.use_openmp:
        sess_options.intra_op_num_threads = 1
    else:
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    logger.info(f"session option: intra_op_num_threads={sess_options.intra_op_num_threads}")

    logger.info(f"Start inferencing onnx model: {onnx_model_path}")
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])

    ort_outputs = onnxruntime_inference(session, input_ids, past, args.total_runs)
    if args.verify_outputs:
        logger.info('PyTorch and OnnxRuntime output 0 (last_state) are close:'.format(0),
                    numpy.allclose(ort_outputs[0], outputs[0].cpu(), rtol=1e-05, atol=1e-04))

        for layer in range(model.config.n_layer):
            logger.info('PyTorch and OnnxRuntime layer {} state (present_{}) are close:'.format(layer, layer),
                        numpy.allclose(ort_outputs[1 + layer], outputs[1][layer].cpu(), rtol=1e-05, atol=1e-04))


if __name__ == '__main__':
    main()
