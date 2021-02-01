import os
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Union
import time
import logging
import numpy
import torch
from transformers import T5ForConditionalGeneration, T5Config

logger = logging.getLogger(__name__)

PRETRAINED_T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3B", "t5-11B"]


class T5Encoder(torch.nn.Module):
    """ T5 encoder outputs only the last hidden state"""
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask)[0]


class T5DecoderNoPastState(torch.nn.Module):
    """ A T5 decoder with LM head"""
    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config
        #self.use_past_state = False

    def forward(self, decoder_input_ids, encoder_hidden_states):
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states)
        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.config.d_model**-0.5)
        lm_logits = self.lm_head(sequence_output)
        return lm_logits


class T5Decoder(torch.nn.Module):
    """ A T5 decoder with LM head and past key values"""
    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config
        #self.use_past_state = True

    def forward(self, decoder_input_ids, decoder_attention_mask, encoder_attention_mask, encoder_hidden_states,
                *past_key_values):
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       attention_mask=decoder_attention_mask,
                                       past_key_values=past_key_values,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_attention_mask)

        sequence_output = decoder_outputs.last_hidden_state
        present_key_values = decoder_outputs.past_key_values

        sequence_output = sequence_output * (self.config.d_model**-0.5)

        lm_logits = self.lm_head(sequence_output)

        return lm_logits, present_key_values


class T5EncoderInputs:
    def __init__(self, input_ids, attention_mask):
        self.input_ids: torch.LongTensor = input_ids
        self.attention_mask: torch.LongTensor = attention_mask

    def to_list(self) -> List:
        input_list = [v for v in [self.input_ids, self.attention_mask] if v is not None]
        return input_list

    def to_tuple(self) -> Tuple:
        return tuple(v for v in [self.input_ids, self.attention_mask] if v is not None)


class T5DecoderInputs:
    def __init__(self, input_ids, attention_mask, encoder_attention_mask, encoder_hidden_states, past_key_values=None):
        self.input_ids: torch.LongTensor = input_ids
        self.attention_mask: torch.LongTensor = attention_mask
        self.encoder_attention_mask: torch.LongTensor = encoder_attention_mask
        self.encoder_hidden_states: Union[torch.FloatTensor, torch.HalfTensor] = encoder_hidden_states
        self.past_key_values: Union[List[List[torch.FloatTensor]], List[List[torch.HalfTensor]], None] = past_key_values

    def to_list(self) -> List:
        input_list = [
            v for v in [self.input_ids, self.attention_mask, self.encoder_attention_mask, self.encoder_hidden_states]
            if v is not None
        ]
        if self.past_key_values:
            input_list.extend(self.past_key_values)
        return input_list

    """
    def to_tuple(self) -> Tuple:
        return tuple(v for v in [
            self.input_ids, self.attention_mask, self.encoder_attention_mask, self.encoder_hidden_states,
            self.past_key_values
        ] if v is not None)

    def to_fp32(self):
        past = [p.to(dtype=torch.float32) for p in self.past_key_values]
        return T5DecoderInputs(self.input_ids, self.attention_mask, self.encoder_attention_mask,
                               self.encoder_hidden_states, past)

    def to_fp16(self):
        past = [p.to(dtype=torch.float16) for p in self.past_key_values]
        return T5DecoderInputs(self.input_ids, self.attention_mask, self.encoder_attention_mask,
                               self.encoder_hidden_states, past)
    """


class IOBindingHelper:
    @staticmethod
    def get_output_buffers(output_shapes, device, is_float16=False):
        """ Returns a dictionary of output name as key, and 1D tensor as value. The tensor has enough space for given shape.
        """
        data_type = torch.float16 if is_float16 else torch.float32

        output_buffers = {}
        for name, shape in output_shapes.items():
            output_buffers[name] = torch.empty(numpy.prod(shape), dtype=data_type, device=device)
        return output_buffers
        
    @staticmethod
    def get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy=True):
        """ Copy results to cpu. Returns a list of numpy array.
        """
        ort_outputs = []
        for output in ort_session.get_outputs():
            output_name = output.name
            buffer = output_buffers[output_name]
            shape = output_shapes[output_name]
            copy_tensor = buffer[0:numpy.prod(shape)].reshape(shape).clone().detach()
            if return_numpy:
                ort_outputs.append(copy_tensor.cpu().numpy())
            else:
                ort_outputs.append(copy_tensor)
        return ort_outputs

class T5EncoderHelper:
    @staticmethod
    def random_inputs(batch_size: int, sequence_length: int, vocab_size: int, device: torch.device) -> T5EncoderInputs:
        """ Create random inputs for T5Encoder. Returns torch tensors of input_ids and attention_mask.

        Args:
            batch_size (int): batch size
            sequence_length (int): sequence length
            vocab_size (int): vocaburary size
            device (torch.device): device of output tensors

        Returns:
            T5EncoderInputs: inputs for encoder
        """
        input_ids = torch.randint(low=0,
                                  high=vocab_size - 1,
                                  size=(batch_size, sequence_length),
                                  dtype=torch.int64,
                                  device=device)

        attention_mask = torch.ones([batch_size, sequence_length], dtype=torch.int64, device=device)
        if sequence_length >= 2:
            # mask one word in a random position
            padding_position = random.randint(0, sequence_length - 1)
            attention_mask[:, padding_position] = 0

        return T5EncoderInputs(input_ids, attention_mask)

    @staticmethod
    def export_onnx(encoder: T5Encoder,
                    device: torch.device,
                    onnx_model_path: str,
                    verbose: bool = True,
                    use_external_data_format: bool = False):
        """Export encoder to ONNX

        Args:
            encoder (T5Encoder): encoder object
            device (torch.device): device of encoder object
            onnx_model_path (str): onnx path
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
        """
        config = encoder.config
        encoder_inputs = T5EncoderHelper.random_inputs(batch_size=1,
                                                       sequence_length=4,
                                                       vocab_size=config.vocab_size,
                                                       device=device)

        with torch.no_grad():
            outputs = encoder(encoder_inputs.input_ids, encoder_inputs.attention_mask)

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(encoder,
                          args=encoder_inputs.to_tuple(),
                          f=onnx_model_path,
                          export_params=True,
                          input_names=['input_ids', 'attention_mask'],
                          output_names=['hidden_states'],
                          example_outputs=outputs,
                          dynamic_axes={
                              'input_ids': {
                                  0: 'batch_size',
                                  1: 'seq_len'
                              },
                              'attention_mask': {
                                  0: 'batch_size',
                                  1: 'seq_len'
                              },
                              'hidden_states': {
                                  0: 'batch_size',
                                  1: 'seq_len'
                              },
                          },
                          opset_version=12,
                          do_constant_folding=True,
                          use_external_data_format=use_external_data_format,
                          verbose=verbose)

    @staticmethod
    def get_output_shapes(batch_size: int, sequence_length: int, config: T5Config) -> Dict[str, List[int]]:
        """ Returns a dictionary with output name as key, and shape as value.
        """
        hidden_size = config.hidden_size
        hidden_state_shape = [batch_size, sequence_length, hidden_size]
        output_shapes = {"hidden_states": hidden_state_shape}
        return output_shapes

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: T5EncoderInputs, total_runs: int = 0):
        """ Run inference of ONNX model, and returns average latency in ms when total_runs > 0 besides outputs.
        """
        logger.debug(f"start onnxruntime_inference")

        ort_inputs = {
            'input_ids': numpy.ascontiguousarray(inputs.input_ids.cpu().numpy()),
            'attention_mask': numpy.ascontiguousarray(inputs.attention_mask.cpu().numpy())
        }

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
    def prepare_io_binding(ort_session, input_ids, attention_mask, output_buffers, output_shapes):
        """ Returnas IO binding object for a session.
        """
        # Bind inputs and outputs to onnxruntime session
        io_binding = ort_session.io_binding()

        # Bind inputs
        assert input_ids.is_contiguous()
        io_binding.bind_input('input_ids', input_ids.device.type, 0, numpy.longlong, list(input_ids.size()),
                              input_ids.data_ptr())

        if attention_mask is not None:
            assert attention_mask.is_contiguous()
            io_binding.bind_input('attention_mask', attention_mask.device.type, 0, numpy.longlong,
                                  list(attention_mask.size()), attention_mask.data_ptr())

        data_type = output_buffers[ort_session.get_outputs()[0].name].dtype
        float_type = numpy.float16 if data_type == torch.float16 else numpy.float32

        # Bind outputs
        for output in ort_session.get_outputs():
            output_name = output.name
            output_buffer = output_buffers[output_name]
            logger.debug(f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}")
            io_binding.bind_output(output_name, output_buffer.device.type, 0, float_type, output_shapes[output_name],
                                   output_buffer.data_ptr())

        return io_binding

    @staticmethod
    def onnxruntime_inference_with_binded_io(ort_session,
                                             inputs: T5EncoderInputs,
                                             output_buffers: Dict[str, torch.Tensor],
                                             output_shapes: Dict[str, List[int]],
                                             total_runs: int = 0,
                                             return_numpy: bool = True,
                                             include_copy_output_latency: bool = False):
        """ Inference with IO binding. Returns outputs, and optional latency when total_runs > 0.
        """
        logger.debug(f"start onnxruntime_inference_with_binded_io")

        # Bind inputs and outputs to onnxruntime session
        io_binding = T5EncoderHelper.prepare_io_binding(ort_session, inputs.input_ids, inputs.attention_mask, output_buffers, output_shapes)

        # Run onnxruntime with io binding
        ort_session.run_with_iobinding(io_binding)

        # Copy results to cpu for verification
        ort_outputs = IOBindingHelper.get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy)

        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            # Run onnxruntime with io binding
            ort_session.run_with_iobinding(io_binding)
            if include_copy_output_latency:
                _ = IOBindingHelper.get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy)
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("OnnxRuntime with IO binding inference time = {} ms".format(format(average_latency, '.2f')))

        return ort_outputs, average_latency

    @staticmethod
    def torchscript(encoder: T5Encoder, device: torch.device):
        """ JIT trace for TorchScript.
        """
        input_list = T5EncoderHelper.random_inputs(batch_size=1,
                                                   sequence_length=1,
                                                   vocab_size=encoder.config.vocab_size,
                                                   device=device).to_list()
        return torch.jit.trace(encoder, input_list)


class T5DecoderHelper:
    @staticmethod
    def random_inputs(decoder:Union[T5DecoderNoPastState, T5Decoder], batch_size: int, sequence_length: int, past_sequence_length: int,  device: torch.device) -> T5DecoderInputs:
        """ Create random inputs for T5Decoder.

        Args:
            decoder: decoder
            batch_size (int): batch size
            sequence_length (int): sequence length
            past sequence_length (int): sequence length for past state
            device (torch.device): device of output tensors
            use_past_state (bool): use past state or not

        Returns:
            T5DecoderInputs: inputs for decoder
        """
        config = decoder.config
        hidden_size: int = config.d_model
        num_attention_heads: int = config.num_heads
        num_layers: int = config.num_layers
        vocab_size: int = config.vocab_size

        input_ids = torch.randint(low=0,
                                  high=vocab_size - 1,
                                  size=(batch_size, sequence_length),
                                  dtype=torch.int64,
                                  device=device)

        if isinstance(decoder, T5DecoderNoPastState):
            #assert past_sequence_length == 0, "past_sequence_length shall be 0 for decoder without past state"
            encoder_hidden_state = torch.rand(batch_size, sequence_length, hidden_size, dtype=torch.float32, device=device)
            return T5DecoderInputs(input_ids, None, None, encoder_hidden_state, None)

        encoder_inputs = T5EncoderHelper.random_inputs(batch_size, past_sequence_length, vocab_size, device)

        encoder_hidden_state = torch.rand(batch_size, past_sequence_length, hidden_size, dtype=torch.float32, device=device)

        attention_mask = torch.ones([batch_size, sequence_length], dtype=torch.int64, device=device)

        past_shape = [batch_size, num_attention_heads, past_sequence_length, int(hidden_size / num_attention_heads)]
        past_one_layer = [torch.rand(past_shape, dtype=torch.float32, device=device) for _ in range(4)]
        past_all_layers = [past_one_layer for _ in range(num_layers)]

        return T5DecoderInputs(input_ids, attention_mask, encoder_inputs.attention_mask, encoder_hidden_state,
                               past_all_layers)

    @staticmethod
    def torchscript(decoder:Union[T5DecoderNoPastState, T5Decoder], device: torch.device):
        """ JIT trace for TorchScript.
        """
        input_list = T5EncoderHelper.random_inputs(decoder,
                                                   batch_size=1,
                                                   sequence_length=1,
                                                   past_sequence_length=1,
                                                   device=device).to_list()
        return torch.jit.trace(decoder, input_list)

    @staticmethod
    def export_onnx(decoder: Union[T5Decoder, T5DecoderNoPastState],
                    device: torch.device,
                    onnx_model_path: str,
                    verbose: bool = True,
                    use_external_data_format: bool = False):
        """Export decoder to ONNX

        Args:
            decoder (UNION[T5Decoder, T5DecoderNoPastState]): decoder object
            device (torch.device): device of decoder object
            onnx_model_path (str): onnx path
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
        """
        assert isinstance(decoder, T5Decoder) or isinstance(decoder, T5DecoderNoPastState)

        config = decoder.config
        num_layers = config.num_layers

        inputs = T5DecoderHelper.random_inputs(decoder,
                                               batch_size=1,
                                               sequence_length=1,
                                               past_sequence_length=1,
                                               device=device)

        input_list = inputs.to_list()
        with torch.no_grad():
            outputs = decoder(*input_list)

        past_names = []
        present_names = []

        if inputs.past_key_values is not None:
            for i in range(num_layers):
                past_names.extend(
                    [f'past_key_self_{i}', f'past_value_self_{i}', f'past_key_cross_{i}', f'past_value_cross_{i}'])
                present_names.extend([
                    f'present_key_self_{i}', f'present_value_self_{i}', f'present_key_cross_{i}',
                    f'present_value_cross_{i}'
                ])

        output_names = ["logits"] + present_names

        # Shape of input tensors:
        #    input_ids: (batch_size, seq_len)
        #    attention_mask: (batch_size, past_seq_len + seq_len)
        #    encoder_attention_mask:
        #    past_*: (batch_size, num_heads, past_seq_len, hidden_size/num_heads)

        # Shape of output tensors:
        #    logits: (batch_size, seq_len, vocab_size)
        #    present_*: (batch_size, num_heads, past_seq_len + seq_len, hidden_size/num_heads)
        input_names = ["input_ids"]
        if inputs.attention_mask is not None:
            input_names.append("attention_mask")
        if inputs.encoder_attention_mask is not None:
            input_names.append("encoder_attention_mask")
        input_names.append("encoder_hidden_states")
        input_names.extend(past_names)

        dynamic_axes = {
            'input_ids': {
                0: 'batch_size',
                1: 'seq_len'
            },
            'attention_mask': {
                0: 'batch_size',
                1: 'seq_len'
            },
            'encoder_attention_mask': {
                0: 'batch_size',
                1: 'past_seq_len'
            },
            'encoder_hidden_states': {
                0: 'batch_size',
                1: 'past_seq_len'
            },
            "logits": {
                0: 'batch_size',
                1: 'seq_len'
            }
        }

        for name in past_names:
            dynamic_axes[name] = {0: 'batch_size', 2: 'past_seq_len'}
        for name in present_names:
            dynamic_axes[name] = {0: 'batch_size', 2: 'total_seq_len'}

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(decoder,
                          args=tuple(input_list),
                          f=onnx_model_path,
                          export_params=True,
                          input_names=input_names,
                          output_names=output_names,
                          example_outputs=outputs,
                          dynamic_axes=dynamic_axes,
                          opset_version=12,
                          do_constant_folding=True,
                          use_external_data_format=use_external_data_format,
                          verbose=verbose)

    @staticmethod
    def get_output_shapes(batch_size: int, sequence_length: int, past_sequence_length: int, config: T5Config,
                          use_past_state: bool) -> Dict[str, List[int]]:
        """ Returns a dictionary with output name as key, and shape as value.
        """

        # Shape of output tensors:
        #    logits: (batch_size, seq_len, vocab_size)
        #    present_*: (batch_size, num_heads, past_seq_len + seq_len, hidden_size/num_heads)
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_heads = config.num_heads
        num_layers = config.num_layers
        logits_shape = [batch_size, sequence_length, vocab_size]

        output_shapes = {"logits": logits_shape}

        if use_past_state:
            for i in range(num_layers):
                present_shape = [batch_size, num_heads, past_sequence_length + sequence_length, hidden_size / num_heads]
                output_shapes[f'present_key_self_{i}'] = present_shape
                output_shapes[f'present_value_self_{i}'] = present_shape
                output_shapes[f'present_key_cross_{i}'] = present_shape
                output_shapes[f'present_value_cross_{i}'] = present_shape

        return output_shapes

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: T5DecoderInputs, total_runs: int = 0):
        """ Run inference of ONNX model, and returns average latency in ms when total_runs > 0 besides outputs.
        """
        logger.debug(f"start onnxruntime_inference")

        ort_inputs = {
            'input_ids': numpy.ascontiguousarray(inputs.input_ids.cpu().numpy()),
            'encoder_hidden_states': numpy.ascontiguousarray(inputs.encoder_hidden_states.cpu().numpy())
        }

        if inputs.attention_mask:
            ort_inputs['attention_mask'] = numpy.ascontiguousarray(inputs.attention_mask.cpu().numpy())

        if inputs.encoder_attention_mask:
            ort_inputs['encoder_attention_mask'] = numpy.ascontiguousarray(inputs.encoder_attention_mask.cpu().numpy())

        if inputs.past_key_values:            
            for i, past_layer_i in enumerate(inputs.past_key_values):
                past_names = [f'past_key_self_{i}', f'past_value_self_{i}', f'past_key_cross_{i}', f'past_value_cross_{i}']
                for j, name in enumerate(past_names):
                    past_tensor = past_layer_i[j]
                    assert past_tensor.is_contiguous()
                    ort_inputs[name] = numpy.ascontiguousarray(past_tensor.cpu().numpy())

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
    def prepare_io_binding(ort_session, inputs: T5DecoderInputs, output_buffers, output_shapes):
        """ Returnas IO binding object for a session.
        """

        # Bind inputs and outputs to onnxruntime session
        io_binding = ort_session.io_binding()

        # Bind inputs
        input_ids = inputs.input_ids
        assert input_ids.is_contiguous()
        io_binding.bind_input('input_ids', input_ids.device.type, 0, numpy.longlong, list(input_ids.size()),
                              input_ids.data_ptr())

        if inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask
            assert attention_mask.is_contiguous()
            io_binding.bind_input('attention_mask', attention_mask.device.type, 0, numpy.longlong, list(attention_mask.size()),
                                attention_mask.data_ptr())

        if inputs.encoder_attention_mask is not None:
            encoder_attention_mask = inputs.encoder_attention_mask
            assert encoder_attention_mask.is_contiguous()
            io_binding.bind_input('encoder_attention_mask', encoder_attention_mask.device.type, 0, numpy.longlong, list(encoder_attention_mask.size()),
                                   encoder_attention_mask.data_ptr())

        data_type = output_buffers[ort_session.get_outputs()[0].name].dtype
        float_type = numpy.float16 if data_type == torch.float16 else numpy.float32

        encoder_hidden_states = inputs.encoder_hidden_states
        assert encoder_hidden_states.is_contiguous()
        io_binding.bind_input('encoder_hidden_states', encoder_hidden_states.device.type, 0, float_type, list(encoder_hidden_states.size()),
                               encoder_hidden_states.data_ptr())

        if inputs.past_key_values is not None:
            for i, past_layer_i in enumerate(inputs.past_key_values):
                past_names = [f'past_key_self_{i}', f'past_value_self_{i}', f'past_key_cross_{i}', f'past_value_cross_{i}']
                for j, name in enumerate(past_names):
                    past_tensor = past_layer_i[j]
                    assert past_tensor.is_contiguous()

                    data_ptr = past_tensor.data_ptr()
                    if data_ptr == 0:
                        # When past_sequence_length is 0, its data_ptr will be zero. IO Binding asserts that data_ptr shall not be zero.
                        # Here we workaround and pass data pointer of input_ids. Actual data is not used for past so it does not matter.
                        data_ptr = input_ids.data_ptr()

                    io_binding.bind_input(name, past_tensor.device.type, 0, float_type, list(past_tensor.size()), data_ptr)

        # Bind outputs
        for output in ort_session.get_outputs():
            output_name = output.name
            output_buffer = output_buffers[output_name]
            logger.debug(f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}")
            io_binding.bind_output(output_name, output_buffer.device.type, 0, float_type, output_shapes[output_name],
                                   output_buffer.data_ptr())

        return io_binding

    @staticmethod
    def onnxruntime_inference_with_binded_io(ort_session,
                                             inputs: T5DecoderInputs,
                                             output_buffers: Dict[str, torch.Tensor],
                                             output_shapes: Dict[str, List[int]],
                                             total_runs: int = 0,
                                             return_numpy: bool = True,
                                             include_copy_output_latency: bool = False):
        """ Inference with IO binding. Returns outputs, and optional latency when total_runs > 0.
        """
        logger.debug(f"start onnxruntime_inference_with_binded_io")

        # Bind inputs and outputs to onnxruntime session
        io_binding = T5DecoderHelper.prepare_io_binding(ort_session, inputs, output_buffers, output_shapes)

        # Run onnxruntime with io binding
        ort_session.run_with_iobinding(io_binding)

        # Copy results to cpu for verification
        ort_outputs = IOBindingHelper.get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes,
                                                                         return_numpy)

        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            # Run onnxruntime with io binding
            ort_session.run_with_iobinding(io_binding)
            if include_copy_output_latency:
                _ = IOBindingHelper.get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes,
                                                                  return_numpy)
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("OnnxRuntime with IO binding inference time = {} ms".format(format(average_latency, '.2f')))

        return ort_outputs, average_latency

class T5Helper:
    @staticmethod
    def get_onnx_path(output_dir: str, model_name_or_path: str, suffix: str = "", new_folder: bool = False):
        """Build onnx path

        Args:
            output_dir (str): output directory
            model_name_or_path (str): pretrained model name, or path to the model checkpoint
            suffix (str, optional): suffix like "_encoder" or "_decoder_fp16" will be appended to file name. Defaults to None.
            new_folder (bool, optional): create a new directory for the model. Defaults to False.

        Returns:
            [type]: [description]
        """
        model_name = model_name_or_path
        if model_name not in PRETRAINED_T5_MODELS and not re.match(
                r'^[\w_-]+$', model_name_or_path):  # It is not a name, shall be a path
            assert os.path.isdir(model_name_or_path)
            model_name = Path(model_name_or_path).parts[-1]

        model_name += suffix

        dir = os.path.join(output_dir, model_name) if new_folder else output_dir
        return os.path.join(dir, model_name + ".onnx")

    @staticmethod
    def load_model(model_name_or_path, cache_dir, device, use_past_state):
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        encoder = T5Encoder(model.encoder, model.config)
        if use_past_state:
            decoder = T5Decoder(model.decoder, model.lm_head, model.config)
        else:
            decoder = T5DecoderNoPastState(model.decoder, model.lm_head, model.config)
        encoder.eval().to(device)
        decoder.eval().to(device)
        return encoder, decoder

    @staticmethod
    def export_onnx(decoder_or_decoder: Union[T5Encoder, T5Decoder, T5DecoderNoPastState],
                    device: torch.device,
                    onnx_model_path: str,
                    verbose: bool = True,
                    use_external_data_format: bool = False):
        if isinstance(decoder_or_decoder, T5Encoder):
            T5EncoderHelper.export_onnx(decoder_or_decoder, device, onnx_model_path, verbose, use_external_data_format)
        else:
            T5DecoderHelper.export_onnx(decoder_or_decoder, device, onnx_model_path, verbose, use_external_data_format)

    @staticmethod
    def optimize_onnx(onnx_model_path,
                      optimized_model_path,
                      is_float16,
                      num_attention_heads,
                      hidden_size,
                      use_external_data_format=False):
        """ Optimize ONNX model with an option to convert it to use mixed precision.
        """
        from optimizer import optimize_model
        m = optimize_model(onnx_model_path,
                           model_type='bert',
                           num_heads=num_attention_heads,
                           hidden_size=hidden_size,
                           opt_level=0,
                           optimization_options=None,
                           use_gpu=False)
        if is_float16:
            m.convert_model_float32_to_float16(cast_input_output=False)

        m.save_model_to_file(optimized_model_path, use_external_data_format)

    @staticmethod
    def pytorch_inference(model, inputs: Union[T5EncoderInputs, T5DecoderInputs], total_runs: int = 0):
        """ Run inference of PyTorch model, and returns average latency in ms when total_runs > 0 besides outputs.
        """
        logger.debug("start pytorch_inference")

        # Convert it to fp32 as the PyTroch model cannot deal with half input.
        #TODO: input_list = inputs.to_fp32().to_list()
        input_list = inputs.to_list()

        with torch.no_grad():
            outputs = model(*input_list)

        if total_runs == 0:
            return outputs

        latency = []
        with torch.no_grad():
            for _ in range(total_runs):
                start = time.time()
                outputs = model(*input_list)
                latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("PyTorch inference time = {} ms".format(format(average_latency, '.2f')))

        return outputs, average_latency