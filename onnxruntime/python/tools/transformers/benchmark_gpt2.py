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
import csv
from datetime import datetime
import psutil
import argparse
import logging
import torch
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig

logger = logging.getLogger('')


# Wrap a class for Onnx model export.
class MyGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, past, attention_mask=None, position_ids=None):
        return super().forward(input_ids,
                               past=past,
                               attention_mask=attention_mask,
                               token_type_ids=None,
                               position_ids=position_ids)


class MyGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, past, attention_mask=None, position_ids=None):
        return super().forward(input_ids,
                               past=past,
                               attention_mask=attention_mask,
                               token_type_ids=None,
                               position_ids=position_ids)


PRETRAINED_MODELS = ['gpt2', 'distilgpt2']

MODEL_CLASSES = ['GPT2LMHeadModel', 'GPT2Model']


def dump_environment():
    if "OMP_NUM_THREADS" in os.environ:
        logger.info("OMP_NUM_THREADS={}".format(os.environ["OMP_NUM_THREADS"]))
    else:
        logger.info("no environment variable of OMP_NUM_THREADS")

    if "OMP_WAIT_POLICY" in os.environ:
        logger.info("OMP_WAIT_POLICY={}".format(os.environ["OMP_WAIT_POLICY"]))
    else:
        logger.info("no environment variable of OMP_WAIT_POLICY")


def setup_environment():
    # ATTENTION: these environment variables must be set before importing onnxruntime.
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
    os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'
    dump_environment()


def pytorch_inference(model, inputs, total_runs=100):
    logger.debug(f"start pytorch_inference")
    input_ids, past, attention_mask, position_ids = inputs

    # Convert it back to fp32 as the PyTroch model cannot deal with half input.
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.float32)
    past = [p.to(dtype=torch.float32) for p in past]

    latency = []
    with torch.no_grad():
        for _ in range(total_runs):
            start = time.time()
            outputs = model(input_ids=input_ids, past=past, attention_mask=attention_mask, position_ids=position_ids)
            latency.append(time.time() - start)

    average_latency = sum(latency) * 1000 / len(latency)
    logger.debug("PyTorch Inference time = {} ms".format(format(average_latency, '.2f')))
    logger.debug(f"PyTorch output 0 shape={outputs[0].shape}")
    logger.debug(f"PyTorch outputs[1][0] shape={outputs[1][0].shape}")
    return outputs, average_latency


def onnxruntime_inference(ort_session, inputs, total_runs=100):
    logger.debug(f"start onnxruntime_inference")
    input_ids, past, attention_mask, position_ids = inputs

    ort_inputs = {'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy())}

    if past is not None:
        for i, past_i in enumerate(past):
            ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past[i].cpu().numpy())

    if attention_mask is not None:
        ort_inputs['attention_mask'] = numpy.ascontiguousarray(attention_mask.cpu().numpy())

    if position_ids is not None:
        ort_inputs['position_ids'] = numpy.ascontiguousarray(position_ids.cpu().numpy())

    latency = []
    for _ in range(total_runs):
        start = time.time()
        ort_outputs = ort_session.run(None, ort_inputs)
        latency.append(time.time() - start)

    average_latency = sum(latency) * 1000 / len(latency)
    logger.debug("OnnxRuntime Inference time = {} ms".format(format(average_latency, '.2f')))

    return ort_outputs, average_latency


def get_dummy_inputs(batch_size, past_sequence_length, sequence_length, num_attention_heads, hidden_size, num_layer,
                     vocab_size, device, use_attention_mask, float16):
    float_type = torch.float16 if float16 else torch.float32
    past_shape = [2, batch_size, num_attention_heads, past_sequence_length, int(hidden_size / num_attention_heads)]
    dummy_past = [torch.rand(past_shape, dtype=float_type, device=device) for _ in range(num_layer)]
    dummy_input_ids = torch.randint(low=0,
                                    high=vocab_size - 1,
                                    size=(batch_size, sequence_length),
                                    dtype=torch.int64,
                                    device=device)
    if use_attention_mask:
        dummy_attention_mask = torch.ones([batch_size, past_sequence_length + sequence_length],
                                          dtype=float_type,
                                          device=device)
        dummy_position_ids = torch.ones([batch_size, sequence_length], dtype=torch.int64,
                                        device=device) * past_sequence_length
        return dummy_input_ids, dummy_past, dummy_attention_mask, dummy_position_ids

    return dummy_input_ids, dummy_past, None, None


def get_output_shapes(batch_size, past_sequence_length, sequence_length, config, use_LMHead):
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_layer = config.n_layer
    vocab_size = config.vocab_size

    last_state_shape = [batch_size, sequence_length, vocab_size
                        ] if use_LMHead else [batch_size, sequence_length, hidden_size]
    present_state_shape = [
        2, batch_size, num_attention_heads, past_sequence_length + sequence_length,
        int(hidden_size / num_attention_heads)
    ]

    output_shapes = {}
    last_state_name = "prediction_scores" if use_LMHead else "last_state"
    output_shapes[last_state_name] = last_state_shape
    for i in range(num_layer):
        output_shapes["present_" + str(i)] = present_state_shape

    return output_shapes


def get_output_buffers(output_shapes, device, is_float16):
    data_type = torch.float16 if is_float16 else torch.float32

    output_buffers = {}
    for name, shape in output_shapes.items():
        output_buffers[name] = torch.empty(numpy.prod(shape), dtype=data_type, device=device)
    return output_buffers


def onnxruntime_inference_with_binded_io(ort_session, inputs, output_buffers, output_shapes, total_runs=100):
    logger.debug(f"start onnxruntime_inference_with_binded_io")
    input_ids, past, attention_mask, position_ids = inputs

    # Bind inputs and outputs to onnxruntime session
    io_binding = ort_session.io_binding()

    # Bind inputs
    io_binding.bind_input('input_ids', input_ids.device.type, 0, numpy.longlong, list(input_ids.size()),
                          input_ids.data_ptr())

    data_type = output_buffers[ort_session.get_outputs()[0].name].dtype
    float_type = numpy.float16 if data_type == torch.float16 else numpy.float32

    if past is not None:
        for i, past_i in enumerate(past):
            io_binding.bind_input(f'past_{i}', past[i].device.type, 0, float_type, list(past[i].size()),
                                  past[i].data_ptr())

    if attention_mask is not None:
        io_binding.bind_input('attention_mask', attention_mask.device.type, 0, float_type, list(attention_mask.size()),
                              attention_mask.data_ptr())

    if position_ids is not None:
        io_binding.bind_input('position_ids', position_ids.device.type, 0, numpy.longlong, list(position_ids.size()),
                              position_ids.data_ptr())

    # Bind outputs
    for output in ort_session.get_outputs():
        output_name = output.name
        output_buffer = output_buffers[output_name]
        logger.debug(f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}")
        io_binding.bind_output(output_name, output_buffer.device.type, 0, float_type, output_shapes[output_name],
                               output_buffer.data_ptr())

    latency = []
    for _ in range(total_runs):
        start = time.time()
        # Run onnxruntime with io binding
        ort_session.run_with_iobinding(io_binding)
        latency.append(time.time() - start)

    average_latency = sum(latency) * 1000 / len(latency)
    logger.debug("OnnxRuntime with IO binding inference time = {} ms".format(format(average_latency, '.2f')))

    # Copy results to cpu for verification
    ort_outputs = []
    for output in ort_session.get_outputs():
        output_name = output.name
        buffer = output_buffers[output_name]
        shape = output_shapes[output_name]
        ort_outputs.append(buffer[0:numpy.prod(shape)].reshape(shape).cpu())

    return ort_outputs, average_latency


def inference(model,
              ort_session,
              inputs,
              output_buffers,
              output_shapes,
              total_runs=100,
              verify_outputs=True,
              disable_ort_io_binding=False):
    outputs, torch_latency = pytorch_inference(model, inputs, total_runs)
    ort_outputs, ort_latency = onnxruntime_inference(ort_session, inputs, total_runs)
    latencies = [torch_latency, ort_latency]
    if verify_outputs:
        logger.info('Verifying Pytorch and ONNX Runtime outputs.')
        verify_ort_outputs(model, outputs, ort_outputs)

    if not disable_ort_io_binding:
        ort_io_outputs, ort_io_latency = onnxruntime_inference_with_binded_io(ort_session, inputs, output_buffers,
                                                                              output_shapes, total_runs)
        latencies.append(ort_io_latency)
        if verify_outputs:
            logger.info('Verifying Pytorch and ONNX Runtime with io binding outputs.')
            verify_ort_outputs(model, outputs, ort_io_outputs)

    return latencies


def verify_ort_outputs(model, torch_outputs, ort_outputs):
    is_close = numpy.allclose(ort_outputs[0], torch_outputs[0].cpu(), rtol=1e-05, atol=1e-04)
    logger.debug(f'PyTorch and OnnxRuntime output 0 (last_state) are close: {is_close}')

    is_all_close = is_close
    for layer in range(model.config.n_layer):
        is_close = numpy.allclose(ort_outputs[1 + layer], torch_outputs[1][layer].cpu(), rtol=1e-05, atol=1e-04)
        logger.debug(f'PyTorch and OnnxRuntime layer {layer} state (present_{layer}) are close:{is_close}')
        is_all_close = is_all_close and is_close

    if not is_all_close:
        logger.warning(f'Failed: PyTorch and OnnxRuntime results are not all close.')
    else:
        logger.info(f'Passed: PyTorch and OnnxRuntime results are all close.')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name',
                        required=True,
                        type=str,
                        choices=PRETRAINED_MODELS,
                        help='Pretrained model selected in the list: ' + ', '.join(PRETRAINED_MODELS))

    parser.add_argument('--model_class',
                        required=False,
                        type=str,
                        default=MODEL_CLASSES[0],
                        choices=MODEL_CLASSES,
                        help='Model type selected in the list: ' + ', '.join(MODEL_CLASSES))

    parser.add_argument('--cache_dir',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'cache_models'),
                        help='Directory to cache pre-trained models')

    parser.add_argument('--onnx_dir',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'onnx_models'),
                        help='Directory to store onnx models')

    parser.add_argument('--test_times',
                        required=False,
                        default=100,
                        type=int,
                        help='Number of repeat times to get average inference latency.')

    parser.add_argument('-v', '--validate_onnx', required=False, action='store_true', help='Validate ONNX model')

    parser.add_argument('-o',
                        '--optimize_onnx',
                        required=False,
                        action='store_true',
                        help='Use optimizer.py to optimize onnx model')
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument('--disable_ort_io_binding',
                        required=False,
                        action='store_true',
                        help='Disable running ONNX Runtime with binded inputs and outputs. ')
    parser.set_defaults(disable_ort_io_binding=False)

    parser.add_argument('--use_attention_mask',
                        required=False,
                        action='store_true',
                        help='Enable attention_mask and position_ids. ')
    parser.set_defaults(use_attention_mask=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--float16', required=False, action='store_true', help="convert model from float32 to float16")
    parser.set_defaults(float16=False)

    parser.add_argument('-b', '--batch_sizes', nargs='+', type=int, default=[1], help="batch size")

    parser.add_argument('-s',
                        '--past_sequence_lengths',
                        nargs='+',
                        type=int,
                        default=[8, 16, 32, 64, 128, 256],
                        help="past sequence lengths")

    parser.add_argument("-r", "--result_csv", required=False, default=None, help="CSV file for saving summary results.")

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
        logging.getLogger("transformers").setLevel(logging.ERROR)
    log_handler.setLevel(logging_level)

    # Avoid duplicated handlers when runing this script in multiple cells of Jupyter Notebook.
    if not logger.hasHandlers():
        logger.addHandler(log_handler)

    logger.setLevel(logging_level)


def export_onnx(model, config, tokenizer, device, output_dir, use_LMHead, use_attention_mask, verbose):
    """ Export GPT-2 model with past state to ONNX model
    """
    num_layer = config.n_layer
    past_names = [f'past_{i}' for i in range(num_layer)]
    present_names = [f'present_{i}' for i in range(num_layer)]

    # GPT2Model outputs last_state; GPT2LMHeadModel outputs prediction_scores
    output_names = ["prediction_scores" if use_LMHead else "last_state"] + present_names

    # Shape of input tensors:
    #    input_ids: (batch_size, seq_len)
    #    past_{i}:  (2, batch_size, num_heads, past_seq_len, hidden_size/num_heads)
    #    attention_mask: (batch_size, past_seq_len + seq_len)
    # Shape of output tensors:
    #    last_state: (batch_size, seq_len, hidden_size)
    #      or prediction_scores: (batch_size, seq_len, vocab_size)
    #    present_{i}:  (2, batch_size, num_heads, past_seq_len + seq_len, hidden_size/num_heads)
    dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'seq_len'}, output_names[0]: {0: 'batch_size', 1: 'seq_len'}}
    for name in past_names:
        dynamic_axes[name] = {1: 'batch_size', 3: 'past_seq_len'}
    for name in present_names:
        dynamic_axes[name] = {1: 'batch_size', 3: 'past_seq_len+seq_len'}

    if use_attention_mask:
        dynamic_axes['attention_mask'] = {0: 'batch_size', 1: 'past_seq_len+seq_len'}
        dynamic_axes['position_ids'] = {0: 'batch_size', 1: 'seq_len'}

    dummy_inputs = get_dummy_inputs(batch_size=1,
                                    past_sequence_length=1,
                                    sequence_length=1,
                                    num_attention_heads=config.num_attention_heads,
                                    hidden_size=config.hidden_size,
                                    num_layer=num_layer,
                                    vocab_size=config.vocab_size,
                                    device=device,
                                    use_attention_mask=use_attention_mask,
                                    float16=False)

    dummy_input_ids, dummy_past, dummy_attention_mask, dummy_position_ids = dummy_inputs

    model_name = "gpt2{}_past{}.onnx".format("_lm" if use_LMHead else "", "_mask" if use_attention_mask else "")
    export_model_path = os.path.join(output_dir, model_name)

    with torch.no_grad():
        outputs = model(input_ids=dummy_input_ids,
                        past=dummy_past,
                        attention_mask=dummy_attention_mask,
                        position_ids=dummy_position_ids)
    logger.info(
        f"Shapes: input_ids={dummy_input_ids.shape} past={dummy_past[0].shape} output={outputs[0].shape} present={outputs[1][0].shape}"
    )

    torch.onnx.export(
        model,
        args=(dummy_input_ids, tuple(dummy_past), dummy_attention_mask, dummy_position_ids) if use_attention_mask else
        (dummy_input_ids, tuple(dummy_past)),
        f=export_model_path,
        input_names=['input_ids'] + past_names + (['attention_mask', 'position_ids'] if use_attention_mask else []),
        output_names=output_names,
        example_outputs=outputs,
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True,
        verbose=verbose)
    return export_model_path


def create_onnxruntime_session(onnx_model_path, use_gpu, verbose):
    session = None
    try:
        from onnxruntime import SessionOptions, InferenceSession
        sess_options = SessionOptions()
        if not use_gpu:
            sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
            logger.debug(f"Session option: intra_op_num_threads={sess_options.intra_op_num_threads}")

        if verbose:
            sess_options.log_severity_level = 0

        logger.debug(f"Create session for onnx model: {onnx_model_path}")
        execution_providers = ['CPUExecutionProvider'
                               ] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = InferenceSession(onnx_model_path, sess_options, providers=execution_providers)
    except:
        logger.error(f"Exception", exc_info=True)

    return session


def main():
    args = parse_arguments()
    setup_logger(args.verbose)

    logger.info(f"Arguments:{args}")
    if args.float16:
        assert args.optimize_onnx and args.use_gpu, "--float16 requires --optimize_onnx --use_gpu"

    dump_environment()

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    output_dir = args.onnx_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_class = MyGPT2LMHeadModel if args.model_class == 'GPT2LMHeadModel' else MyGPT2Model
    use_LMHead = (args.model_class == 'GPT2LMHeadModel')

    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name, torchscript=False, cache_dir=cache_dir)
    model = model_class.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    # This scirpt does not support float16 for PyTorch.
    #if args.float16:
    #    model.half()

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.to(device)
    export_model_path = export_onnx(model, config, tokenizer, device, output_dir, use_LMHead, args.use_attention_mask,
                                    args.verbose)

    # setup environment variables before importing onnxruntime.
    setup_environment()
    import onnxruntime

    if args.use_gpu:
        assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(
        ), "Please install onnxruntime-gpu package to test GPU inference."

    if not args.optimize_onnx:
        onnx_model_path = export_model_path
    else:
        from optimizer import optimize_model
        m = optimize_model(export_model_path,
                           model_type='gpt2',
                           num_heads=config.num_attention_heads,
                           hidden_size=config.hidden_size,
                           opt_level=0,
                           optimization_options=None,
                           use_gpu=args.use_gpu)
        if args.float16:
            m.convert_model_float32_to_float16(cast_input_output=False)

        filename_prefix = "gpt2{}_past{}".format("_lm" if use_LMHead else "",
                                                 "_mask" if args.use_attention_mask else "")
        onnx_model_path = os.path.join(
            output_dir, '{}_{}_{}.onnx'.format(filename_prefix, "gpu" if args.use_gpu else "cpu",
                                               "fp16" if args.float16 else "fp32"))
        m.save_model_to_file(onnx_model_path)

    session = create_onnxruntime_session(onnx_model_path, args.use_gpu, args.verbose)
    if session is None:
        return

    # One word is generated for each inference. This length does not include that of past state.
    sequence_length = 1

    # Allocate output buffers for IO Binding
    output_buffers = {}
    if not args.disable_ort_io_binding:
        max_output_shapes = get_output_shapes(max(args.batch_sizes), max(args.past_sequence_lengths), sequence_length,
                                              config, use_LMHead)
        output_buffers = get_output_buffers(max_output_shapes, device, args.float16)

    csv_filename = args.result_csv or "benchmark_result_{}.csv".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "model_name", "model_class", "gpu", "fp16", "use_attention_mask", "optimizer", "io_binding", "batch_size",
            "past_sequence_length", "torch_latency", "ort_latency", "ort_io_latency"
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        for batch_size in args.batch_sizes:
            for past_sequence_length in args.past_sequence_lengths:
                logger.debug(f"Running test for batch_size={batch_size} past_sequence_length={past_sequence_length}...")
                dummy_inputs = get_dummy_inputs(batch_size, past_sequence_length, sequence_length,
                                                config.num_attention_heads, config.hidden_size, config.n_layer,
                                                config.vocab_size, device, args.use_attention_mask, args.float16)
                output_shapes = get_output_shapes(batch_size, past_sequence_length, sequence_length, config, use_LMHead)

                try:
                    latencies = inference(model,
                                          session,
                                          dummy_inputs,
                                          output_buffers,
                                          output_shapes,
                                          args.test_times,
                                          verify_outputs=args.validate_onnx,
                                          disable_ort_io_binding=args.disable_ort_io_binding)
                    ort_io_latency_info = f", ort_io_latency={latencies[2]:.2f}" if not args.disable_ort_io_binding else ""
                    logger.info(
                        f"batch_size={batch_size}, past_sequence_length={past_sequence_length}, torch_latency={latencies[0]:.2f}, ort_latency={latencies[1]:.2f}{ort_io_latency_info}"
                    )

                    row = {
                        "model_name": args.model_name,
                        "model_class": args.model_class,
                        "gpu": args.use_gpu,
                        "fp16": args.float16,
                        "use_attention_mask": args.use_attention_mask,
                        "optimizer": args.optimize_onnx,
                        "io_binding": not args.disable_ort_io_binding,
                        "batch_size": batch_size,
                        "past_sequence_length": past_sequence_length,
                        "torch_latency": f"{latencies[0]:.2f}",
                        "ort_latency": f"{latencies[1]:.2f}",
                        "ort_io_latency": f"{latencies[2]:.2f}" if not args.disable_ort_io_binding else ""
                    }
                    csv_writer.writerow(row)
                except:
                    logger.error(f"Exception", exc_info=True)
    logger.info(f"Results are saved to file {csv_filename}")


if __name__ == '__main__':
    main()
