import re

import numpy as np
from onnx.onnx_ml_pb2 import TensorShapeProto
import torch
import time
import multiprocessing
from .base import BaseModelWrapper
from onnx import numpy_helper
import myutils

def load_onnx(onnx_path, device):
    '''Load ONNX model'''
    import onnxruntime as ort
    if device.lower() == 'cpu':
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        execution_providers = ['CPUExecutionProvider']
        ort_sess = ort.InferenceSession(onnx_path, sess_options, providers=execution_providers)
    else:
        execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_sess = ort.InferenceSession(onnx_path, providers=execution_providers)
    return ort_sess


class OnnxModelWrapper(BaseModelWrapper):
    '''
    Arguments
    =========
    ort_sess: onnxruntime InferenceSession
    '''
    type_lookups = {torch.float: np.float32,
                    torch.float16: np.float16,
                    torch.long: np.longlong,
                    'tensor(float16)': torch.float16,
                    'tensor(float)': torch.float}

    def __init__(self, ort_sess, pad_token_id=0, io_binding=True, max_batch_size=16, max_sequence_length=64):
        super().__init__(ort_sess, pad_token_id=pad_token_id, float_type=None)
        self.ort_sess = ort_sess
        past_inputs = [x for x in self.ort_sess.get_inputs() if x.name.startswith('past')]
        self.float_type = torch.float16 if 'float16' in past_inputs[0].type else torch.float32
        self.n_layers = len(past_inputs)
        self.past_shape = past_inputs[0].shape
        self.io_binding = io_binding
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length

        ort_providers = ort_sess.get_providers()
        self.device = torch.device('cuda' if any('cuda' in x.lower() for x in ort_providers) else 'cpu')
        if self.io_binding:
            self.init_buffers()

    def convert_input_for_onnx(self, model_inputs):
        past = model_inputs.pop('past') if 'past' in model_inputs else None
        if past is not None:
            model_inputs.update({f'past_{i}': p for i, p in enumerate(past)})
        else:
            shape = [x for x in self.past_shape]
            shape[3] = 0
            shape[1] = model_inputs['input_ids'].shape[0]
            zeros = torch.zeros(shape).type_as(model_inputs['attention_mask'])
            model_inputs.update({f'past_{i}': zeros for i in range(self.n_layers)})

        if self.io_binding:
            model_inputs = {k: v.contiguous() for k, v in model_inputs.items()}
        else:
            model_inputs = {k: v.cpu().numpy() for k, v in model_inputs.items()}
        return model_inputs

    def forward(self, input_ids, model_state, generator_state, is_beam_search=True):
        """
        :param input_ids: [LongTensor(shape=(batch_size, seq_len)). Here batch_size means `num_contexts x num_sequence_per_sample`
        :param model_state: the `past` in `GPT2LMHeadModel` call.
        """
        _, _, cur_len = input_ids.shape
        input_ids = input_ids.view(-1, cur_len)

        device = input_ids.device
        model_inputs = self.get_model_inputs(input_ids, model_state, generator_state)
        model_inputs = self.convert_input_for_onnx(model_inputs)

        if self.io_binding:
            start_time = time.perf_counter()
            outputs = self.run_with_io_binding(model_inputs)
            end_time = time.perf_counter()
            infer_time = (end_time - start_time) * 1000
            myutils.total_infer_time += infer_time
        else:
            device = input_ids.device
            outputs = self.ort_sess.run(None, model_inputs)  # list outputs
            outputs = [torch.from_numpy(x).to(device) for x in outputs]

        next_token_logits = outputs[0]
        present = outputs[1:]
        return next_token_logits, present

    def init_buffers(self):
        size = self.max_batch_size * self.max_sequence_length
        self.output_buffers = {}
        for output in self.ort_sess.get_outputs():
            shape = list(output.shape)
            shape[shape.index('batch_size')] = 1
            idx = [i for i, s in enumerate(shape) if isinstance(s, str) and 'seq' in s][0]
            shape[idx] = size
            self.output_buffers[output.name] = torch.zeros(np.prod(shape), dtype=self.type_lookups[output.type], device=self.device)
        self.buffer_sizes = {k: v.size(0) for k, v in self.output_buffers.items()}

    def get_output_shape(self, input_dict):
        vocab_size = self.ort_sess.get_outputs()[0].shape[-1]
        logits_shape = tuple(input_dict['input_ids'].size()) + (vocab_size,)
        present_shape = list(input_dict['past_0'].size())
        present_shape[3] += logits_shape[1]
        output_shapes = {x.name: present_shape for x in self.ort_sess.get_outputs() if x.name.startswith('present')}
        output_shapes['logits'] = logits_shape
        return output_shapes

    def get_outputs_from_buffers(self, buffers, output_shapes):
        outputs = {}
        for name, shape in output_shapes.items():
            output = buffers[name].view(-1)[:np.prod(shape)].view(shape)
            outputs[name] = output
        return outputs

    def run_with_io_binding(self, input_dict):
        # Bind inputs and outputs to onnxruntime session
        io_binding = self.ort_sess.io_binding()

        # the inputs need to be contiguous
        input_dict = {k: v.contiguous() for k, v in input_dict.items()}

        # as a fake pointer for empty tensors
        input_ids_ptr = input_dict['input_ids'].data_ptr()

        for n, tensor in input_dict.items():
            data_type = self.type_lookups[tensor.dtype]
            data_ptr = tensor.data_ptr() if tensor.data_ptr() != 0 else input_ids_ptr
            io_binding.bind_input(n, tensor.device.type, 0, data_type, list(tensor.size()), data_ptr)

        output_shapes = self.get_output_shape(input_dict)
        ort_output_info = self.ort_sess.get_outputs()
        for output in ort_output_info:
            output_name = output.name
            output_buffer = self.output_buffers[output_name]
            data_type = self.type_lookups[output_buffer.dtype]
            io_binding.bind_output(output_name, output_buffer.device.type, 0, data_type, output_shapes[output_name],
                                   output_buffer.data_ptr())

        self.ort_sess.run_with_iobinding(io_binding)

        outputs = self.get_outputs_from_buffers(self.output_buffers, output_shapes)
        return [outputs[output.name] for output in ort_output_info]
