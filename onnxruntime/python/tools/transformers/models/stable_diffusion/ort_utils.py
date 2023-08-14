# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import OrderedDict
from typing import Dict

import torch

import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import TypeHelper


class OrtCudaSession:
    """ONNX Runtime Session for CUDA or TensorRT provider"""

    def __init__(self, ort_session: ort.InferenceSession, device: torch.device, enable_cuda_graph=False):
        self.ort_session = ort_session
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        self.io_name_to_numpy_type = TypeHelper.get_io_numpy_type_map(self.ort_session)
        self.io_binding = self.ort_session.io_binding()
        self.enable_cuda_graph = enable_cuda_graph

        self.input_tensors = OrderedDict()
        self.output_tensors = OrderedDict()
        self.device = device

    def __del__(self):
        del self.input_tensors
        del self.output_tensors
        del self.io_binding
        del self.ort_session

    def allocate_buffers(self, shape_dict: Dict[str, tuple]):
        """Allocate tensors for I/O Binding"""
        if self.enable_cuda_graph:
            for name, shape in shape_dict.items():
                if name in self.input_names:
                    # Reuse allocated buffer when the shape is same
                    if name in self.input_tensors:
                        if tuple(self.input_tensors[name].shape) == tuple(shape):
                            continue
                        raise RuntimeError("Expect static input shape for cuda graph")

                    numpy_dtype = self.io_name_to_numpy_type[name]
                    tensor = torch.empty(tuple(shape), dtype=TypeHelper.numpy_type_to_torch_type(numpy_dtype)).to(
                        device=self.device
                    )
                    self.input_tensors[name] = tensor

                    self.io_binding.bind_input(
                        name,
                        tensor.device.type,
                        tensor.device.index,
                        numpy_dtype,
                        list(tensor.size()),
                        tensor.data_ptr(),
                    )

        for name, shape in shape_dict.items():
            if name in self.output_names:
                # Reuse allocated buffer when the shape is same
                if name in self.output_tensors and tuple(self.output_tensors[name].shape) == tuple(shape):
                    continue

                numpy_dtype = self.io_name_to_numpy_type[name]
                tensor = torch.empty(tuple(shape), dtype=TypeHelper.numpy_type_to_torch_type(numpy_dtype)).to(
                    device=self.device
                )
                self.output_tensors[name] = tensor

                self.io_binding.bind_output(
                    name,
                    tensor.device.type,
                    tensor.device.index,
                    numpy_dtype,
                    list(tensor.size()),
                    tensor.data_ptr(),
                )

    def infer(self, feed_dict):
        """Bind input tensors and run inference"""
        for name, tensor in feed_dict.items():
            assert isinstance(tensor, torch.Tensor) and tensor.is_contiguous()
            if name in self.input_names:
                if self.enable_cuda_graph:
                    assert self.input_tensors[name].nelement() == tensor.nelement()
                    assert tensor.device.type == "cuda"
                    # Update input tensor inplace since cuda graph requires input and output has fixed memory address.
                    from cuda import cudart

                    cudart.cudaMemcpy(
                        self.input_tensors[name].data_ptr(),
                        tensor.data_ptr(),
                        tensor.element_size() * tensor.nelement(),
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    )
                else:
                    self.io_binding.bind_input(
                        name,
                        tensor.device.type,
                        tensor.device.index,
                        TypeHelper.torch_type_to_numpy_type(tensor.dtype),
                        [1] if len(tensor.shape) == 0 else list(tensor.shape),
                        tensor.data_ptr(),
                    )

        self.ort_session.run_with_iobinding(self.io_binding)

        return self.output_tensors
