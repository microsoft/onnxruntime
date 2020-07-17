# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script benchmarks gpt2 model with past state.
# For gpt2 model without past state, use benchmark.py to measure performance.
import os
import logging
import torch
import onnx
import random
import numpy
import time
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from benchmark_helper import Precision

logger = logging.getLogger('')

DEFAULT_TOLERANCE = {Precision.FLOAT32: 0.0005, Precision.FLOAT16: 0.2, Precision.INT8: 3.0}


# Here we wrap a class to disable past state output.
class GPT2ModelNoPastState(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids):
        return super().forward(input_ids, use_cache=False)


# Wrap a class for Onnx model export.
class MyGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, position_ids, attention_mask, *past):
        return super().forward(input_ids, position_ids=position_ids, attention_mask=attention_mask, past=past)


class MyGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, position_ids, attention_mask, *past):
        return super().forward(input_ids, position_ids=position_ids, attention_mask=attention_mask, past=past)


MODEL_CLASSES = {'GPT2LMHeadModel': (MyGPT2LMHeadModel, 'logits'), 'GPT2Model': (MyGPT2Model, 'last_state')}


class Gpt2Helper:
    @staticmethod
    def get_dummy_inputs(batch_size, past_sequence_length, sequence_length, num_attention_heads, hidden_size, num_layer,
                         vocab_size, device, float16):
        float_type = torch.float16 if float16 else torch.float32
        past_shape = [2, batch_size, num_attention_heads, past_sequence_length, int(hidden_size / num_attention_heads)]

        past = [torch.rand(past_shape, dtype=float_type, device=device) for _ in range(num_layer)]
        input_ids = torch.randint(low=0,
                                  high=vocab_size - 1,
                                  size=(batch_size, sequence_length),
                                  dtype=torch.int64,
                                  device=device)

        total_sequence_length = past_sequence_length + sequence_length
        attention_mask = torch.ones([batch_size, total_sequence_length], dtype=float_type, device=device)
        if total_sequence_length >= 2:
            padding_position = random.randint(0, total_sequence_length - 1)  # test input with padding.
            attention_mask[:, padding_position] = 0

        # Deduce position_ids from attention mask
        position_ids = (attention_mask.long().cumsum(-1) - 1)
        position_ids.masked_fill_(position_ids < 0, 0)
        position_ids = position_ids[:, past_sequence_length:]

        return input_ids, position_ids, attention_mask, past

    @staticmethod
    def get_output_shapes(batch_size, past_sequence_length, sequence_length, config, model_class):
        num_attention_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        num_layer = config.num_hidden_layers
        vocab_size = config.vocab_size

        output_name = MODEL_CLASSES[model_class][1]

        last_state_shape = [
            batch_size, sequence_length, vocab_size if model_class == "GPT2LMHeadModel" else hidden_size
        ]
        present_state_shape = [
            2, batch_size, num_attention_heads, past_sequence_length + sequence_length,
            int(hidden_size / num_attention_heads)
        ]

        output_shapes = {output_name: last_state_shape}
        for i in range(num_layer):
            output_shapes["present_" + str(i)] = present_state_shape

        return output_shapes

    @staticmethod
    def get_output_buffers(output_shapes, device, is_float16):
        data_type = torch.float16 if is_float16 else torch.float32

        output_buffers = {}
        for name, shape in output_shapes.items():
            output_buffers[name] = torch.empty(numpy.prod(shape), dtype=data_type, device=device)
        return output_buffers

    @staticmethod
    def diff_outputs(torch_outputs, ort_outputs, relative=False):
        expected_outputs = torch_outputs[0].cpu().numpy()
        diff = numpy.abs(expected_outputs - ort_outputs[0])
        if relative:
            return numpy.amax(diff / (numpy.abs(expected_outputs) + 1e-6))
        else:
            return numpy.amax(diff)

    @staticmethod
    def compare_outputs(torch_outputs, ort_outputs, rtol=1e-03, atol=1e-03):
        is_close = numpy.allclose(ort_outputs[0], torch_outputs[0].cpu(), rtol=rtol, atol=atol)
        logger.debug(f'PyTorch and OnnxRuntime output 0 (last_state) are close: {is_close}')

        is_all_close = is_close
        num_layers = len(ort_outputs) - 1
        for layer in range(num_layers):
            is_close = numpy.allclose(ort_outputs[1 + layer], torch_outputs[1][layer].cpu(), rtol=rtol, atol=atol)
            logger.debug(f'PyTorch and OnnxRuntime layer {layer} state (present_{layer}) are close:{is_close}')
            is_all_close = is_all_close and is_close

        if not is_all_close:
            max_abs_diff = Gpt2Helper.diff_outputs(torch_outputs, ort_outputs)
            logger.info(f'PyTorch and OnnxRuntime results are not all close: max_abs_diff={max_abs_diff:.5f}')

        return is_all_close

    @staticmethod
    def export_onnx(model, device, onnx_model_path, verbose=False):
        """ Export GPT-2 model with past state to ONNX model
        """
        config: GPT2Config = model.config
        num_layer = config.n_layer
        dummy_inputs = Gpt2Helper.get_dummy_inputs(batch_size=1,
                                                   past_sequence_length=1,
                                                   sequence_length=1,
                                                   num_attention_heads=config.num_attention_heads,
                                                   hidden_size=config.hidden_size,
                                                   num_layer=num_layer,
                                                   vocab_size=config.vocab_size,
                                                   device=device,
                                                   float16=False)

        dummy_input_ids, dummy_position_ids, dummy_attention_mask, dummy_past = dummy_inputs

        input_list = [dummy_input_ids, dummy_position_ids, dummy_attention_mask] + dummy_past
        with torch.no_grad():
            outputs = model(*input_list)

        past_names = [f'past_{i}' for i in range(num_layer)]
        present_names = [f'present_{i}' for i in range(num_layer)]

        # GPT2Model outputs last_state; GPT2LMHeadModel outputs logits (prediction_scores)
        assert outputs[0].shape[2] == config.vocab_size or outputs[0].shape[2] == config.hidden_size
        output_names = ["logits" if outputs[0].shape[2] == config.vocab_size else "last_state"] + present_names

        # Shape of input tensors:
        #    input_ids: (batch_size, seq_len)
        #    past_{i}:  (2, batch_size, num_heads, past_seq_len, hidden_size/num_heads)
        #    attention_mask: (batch_size, past_seq_len + seq_len)
        # Shape of output tensors:
        #    last_state: (batch_size, seq_len, hidden_size)
        #      or logits: (batch_size, seq_len, vocab_size)
        #    present_{i}:  (2, batch_size, num_heads, past_seq_len + seq_len, hidden_size/num_heads)
        dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'seq_len'}, output_names[0]: {0: 'batch_size', 1: 'seq_len'}}
        for name in past_names:
            dynamic_axes[name] = {1: 'batch_size', 3: 'past_seq_len'}
        for name in present_names:
            dynamic_axes[name] = {1: 'batch_size', 3: 'total_seq_len'}

        dynamic_axes['attention_mask'] = {0: 'batch_size', 1: 'total_seq_len'}
        dynamic_axes['position_ids'] = {0: 'batch_size', 1: 'seq_len'}

        logger.info(
            f"Shapes: input_ids={dummy_input_ids.shape} past={dummy_past[0].shape} output={outputs[0].shape} present={outputs[1][0].shape}"
        )

        torch.onnx.export(model,
                          args=tuple(input_list),
                          f=onnx_model_path,
                          input_names=['input_ids', 'position_ids', 'attention_mask'] + past_names,
                          output_names=output_names,
                          example_outputs=outputs,
                          dynamic_axes=dynamic_axes,
                          opset_version=11,
                          do_constant_folding=True,
                          verbose=verbose)

    def optimize_onnx(onnx_model_path, optimized_model_path, is_float16, num_attention_heads, hidden_size):
        from optimizer import optimize_model
        m = optimize_model(onnx_model_path,
                           model_type='gpt2',
                           num_heads=num_attention_heads,
                           hidden_size=hidden_size,
                           opt_level=0,
                           optimization_options=None,
                           use_gpu=False)
        if is_float16:
            m.convert_model_float32_to_float16(cast_input_output=False)

        m.save_model_to_file(optimized_model_path)

    @staticmethod
    def pytorch_inference(model, inputs, total_runs=0):
        logger.debug(f"start pytorch_inference")
        input_ids, position_ids, attention_mask, past = inputs

        # Convert it back to fp32 as the PyTroch model cannot deal with half input.
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float32)
        past = [p.to(dtype=torch.float32) for p in past]

        input_list = [input_ids, position_ids, attention_mask] + past

        with torch.no_grad():
            outputs = model(*input_list)

        if total_runs == 0:
            return ort_outputs

        latency = []
        with torch.no_grad():
            for _ in range(total_runs):
                start = time.time()
                outputs = model(*input_list)
                latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("PyTorch inference time = {} ms".format(format(average_latency, '.2f')))

        return outputs, average_latency

    @staticmethod
    def onnxruntime_inference(ort_session, inputs, total_runs=0):
        logger.debug(f"start onnxruntime_inference")
        input_ids, position_ids, attention_mask, past = inputs

        ort_inputs = {'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy())}

        if past is not None:
            for i, past_i in enumerate(past):
                ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past[i].cpu().numpy())

        if attention_mask is not None:
            ort_inputs['attention_mask'] = numpy.ascontiguousarray(attention_mask.cpu().numpy())

        if position_ids is not None:
            ort_inputs['position_ids'] = numpy.ascontiguousarray(position_ids.cpu().numpy())

        ort_outputs = ort_session.run(None, ort_inputs)
        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            ort_outputs = ort_session.run(None, ort_inputs)
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("OnnxRuntime Inference time = {} ms".format(format(average_latency, '.2f')))

        return ort_outputs, average_latency

    @staticmethod
    def onnxruntime_inference_with_binded_io(ort_session, inputs, output_buffers, output_shapes, total_runs=0):
        logger.debug(f"start onnxruntime_inference_with_binded_io")
        input_ids, position_ids, attention_mask, past = inputs

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
            io_binding.bind_input('attention_mask', attention_mask.device.type, 0, float_type,
                                  list(attention_mask.size()), attention_mask.data_ptr())

        if position_ids is not None:
            io_binding.bind_input('position_ids', position_ids.device.type, 0, numpy.longlong,
                                  list(position_ids.size()), position_ids.data_ptr())

        # Bind outputs
        for output in ort_session.get_outputs():
            output_name = output.name
            output_buffer = output_buffers[output_name]
            logger.debug(f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}")
            io_binding.bind_output(output_name, output_buffer.device.type, 0, float_type, output_shapes[output_name],
                                   output_buffer.data_ptr())

        # Copy results to cpu for verification
        ort_outputs = []
        for output in ort_session.get_outputs():
            output_name = output.name
            buffer = output_buffers[output_name]
            shape = output_shapes[output_name]
            ort_outputs.append(buffer[0:numpy.prod(shape)].reshape(shape).cpu())

        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            # Run onnxruntime with io binding
            ort_session.run_with_iobinding(io_binding)
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("OnnxRuntime with IO binding inference time = {} ms".format(format(average_latency, '.2f')))

        return ort_outputs, average_latency

    @staticmethod
    def test_parity(ort_session, model, device, is_float16=False, rtol=5e-4, atol=5e-4, total_test_cases=100):
        config: GPT2Config = model.config

        passed_test_cases = 0
        for _ in range(total_test_cases):
            sequence_length = random.randint(1, 32)
            past_sequence_length = random.randint(0, 128)
            batch_size = random.randint(1, 16)

            logger.debug(
                f"Running parity test for batch_size={batch_size} past_sequence_length={past_sequence_length}...")
            dummy_inputs = Gpt2Helper.get_dummy_inputs(batch_size, past_sequence_length, sequence_length,
                                                       config.num_attention_heads, config.hidden_size, config.n_layer,
                                                       config.vocab_size, device, is_float16)
            outputs = Gpt2Helper.pytorch_inference(model, dummy_inputs)
            ort_outputs = Gpt2Helper.onnxruntime_inference(ort_session, dummy_inputs)
            is_all_close = Gpt2Helper.compare_outputs(outputs, ort_outputs, rtol=rtol, atol=atol)
            if is_all_close:
                passed_test_cases += 1
        logger.info(f"Parity Test Cases={total_test_cases}; Passed={passed_test_cases} (rtol={rtol}, atol={atol})")
        if passed_test_cases > 0.95 * total_test_cases:
            logger.info(f"Parity is good: passed rate={int(passed_test_cases*100/total_test_cases):.0f}%")
        return passed_test_cases == total_test_cases

    @staticmethod
    def torchscript(model, config, device):
        dummy_inputs = Gpt2Helper.get_dummy_inputs(batch_size=1,
                                                   past_sequence_length=1,
                                                   sequence_length=1,
                                                   num_attention_heads=config.num_attention_heads,
                                                   hidden_size=config.hidden_size,
                                                   num_layer=config.n_layer,
                                                   vocab_size=config.vocab_size,
                                                   device=device,
                                                   float16=False)
        dummy_input_ids, dummy_position_ids, dummy_attention_mask, dummy_past = dummy_inputs
        return torch.jit.trace(model, [dummy_input_ids, dummy_position_ids, dummy_attention_mask] + dummy_past)

    @staticmethod
    def get_onnx_paths(output_dir, model_name_or_path, model_class: str = 'GPT2LMHeadModel', has_past=True):
        model_name = model_name_or_path if model_name_or_path.isalnum() else os.path.dirname(model_name_or_path)

        if model_class != 'GPT2LMHeadModel':
            model_name += "_" + model_class

        if has_past:
            model_name += "_past"

        return {
            "raw": os.path.join(output_dir, model_name + ".onnx"),
            "fp32": os.path.join(output_dir, model_name + "_fp32.onnx"),
            "fp16": os.path.join(output_dir, model_name + "_fp16.onnx"),
            "int8": os.path.join(output_dir, model_name + "_int8.onnx")
        }
