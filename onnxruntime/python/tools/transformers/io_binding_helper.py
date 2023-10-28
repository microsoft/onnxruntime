import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy
import torch

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


class TypeHelper:
    @staticmethod
    def get_input_type(ort_session: InferenceSession, name: str) -> str:
        for _i, input in enumerate(ort_session.get_inputs()):
            if input.name == name:
                return input.type
        raise ValueError(f"input name {name} not found")

    @staticmethod
    def get_output_type(ort_session, name: str) -> str:
        for _i, output in enumerate(ort_session.get_outputs()):
            if output.name == name:
                return output.type

        raise ValueError(f"output name {name} not found")

    @staticmethod
    def ort_type_to_numpy_type(ort_type: str):
        ort_type_to_numpy_type_map = {
            "tensor(int64)": numpy.longlong,
            "tensor(int32)": numpy.intc,
            "tensor(float)": numpy.float32,
            "tensor(float16)": numpy.float16,
            "tensor(bool)": bool,
        }
        if ort_type not in ort_type_to_numpy_type_map:
            raise ValueError(f"{ort_type} not found in map")

        return ort_type_to_numpy_type_map[ort_type]

    @staticmethod
    def ort_type_to_torch_type(ort_type: str):
        ort_type_to_torch_type_map = {
            "tensor(int64)": torch.int64,
            "tensor(int32)": torch.int32,
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
            "tensor(bool)": torch.bool,
        }
        if ort_type not in ort_type_to_torch_type_map:
            raise ValueError(f"{ort_type} not found in map")

        return ort_type_to_torch_type_map[ort_type]

    @staticmethod
    def numpy_type_to_torch_type(numpy_type: numpy.dtype):
        numpy_type_to_torch_type_map = {
            numpy.longlong: torch.int64,
            numpy.intc: torch.int32,
            numpy.int32: torch.int32,
            numpy.float32: torch.float32,
            numpy.float16: torch.float16,
            bool: torch.bool,
        }
        if numpy_type not in numpy_type_to_torch_type_map:
            raise ValueError(f"{numpy_type} not found in map")

        return numpy_type_to_torch_type_map[numpy_type]

    @staticmethod
    def torch_type_to_numpy_type(torch_type: torch.dtype):
        torch_type_to_numpy_type_map = {
            torch.int64: numpy.longlong,
            torch.int32: numpy.intc,
            torch.float32: numpy.float32,
            torch.float16: numpy.float16,
            torch.bool: bool,
        }
        if torch_type not in torch_type_to_numpy_type_map:
            raise ValueError(f"{torch_type} not found in map")

        return torch_type_to_numpy_type_map[torch_type]

    @staticmethod
    def get_io_numpy_type_map(ort_session: InferenceSession) -> Dict[str, numpy.dtype]:
        """Create a mapping from input/output name to numpy data type"""
        name_to_numpy_type = {}
        for input in ort_session.get_inputs():
            name_to_numpy_type[input.name] = TypeHelper.ort_type_to_numpy_type(input.type)

        for output in ort_session.get_outputs():
            name_to_numpy_type[output.name] = TypeHelper.ort_type_to_numpy_type(output.type)
        return name_to_numpy_type


class IOBindingHelper:
    @staticmethod
    def get_output_buffers(ort_session: InferenceSession, output_shapes, device):
        """Returns a dictionary of output name as key, and 1D tensor as value. The tensor has enough space for given shape."""
        output_buffers = {}
        for name, shape in output_shapes.items():
            ort_type = TypeHelper.get_output_type(ort_session, name)
            torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
            output_buffers[name] = torch.empty(numpy.prod(shape), dtype=torch_type, device=device)
        return output_buffers

    @staticmethod
    def prepare_io_binding(
        ort_session,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past: List[torch.Tensor],
        output_buffers,
        output_shapes,
        name_to_np_type=None,
    ):
        """Returnas IO binding object for a session."""
        if name_to_np_type is None:
            name_to_np_type = TypeHelper.get_io_numpy_type_map(ort_session)

        # Bind inputs and outputs to onnxruntime session
        io_binding = ort_session.io_binding()

        # Bind inputs
        assert input_ids.is_contiguous()
        io_binding.bind_input(
            "input_ids",
            input_ids.device.type,
            0,
            name_to_np_type["input_ids"],
            list(input_ids.size()),
            input_ids.data_ptr(),
        )

        if past is not None:
            for i, past_i in enumerate(past):
                assert past_i.is_contiguous()

                data_ptr = past_i.data_ptr()
                if data_ptr == 0:
                    # When past_sequence_length is 0, its data_ptr will be zero. IO Binding asserts that data_ptr shall not be zero.
                    # Here we workaround and pass data pointer of input_ids. Actual data is not used for past so it does not matter.
                    data_ptr = input_ids.data_ptr()

                io_binding.bind_input(
                    f"past_{i}",
                    past_i.device.type,
                    0,
                    name_to_np_type[f"past_{i}"],
                    list(past_i.size()),
                    data_ptr,
                )

        if attention_mask is not None:
            assert attention_mask.is_contiguous()
            io_binding.bind_input(
                "attention_mask",
                attention_mask.device.type,
                0,
                name_to_np_type["attention_mask"],
                list(attention_mask.size()),
                attention_mask.data_ptr(),
            )

        if position_ids is not None:
            assert position_ids.is_contiguous()
            io_binding.bind_input(
                "position_ids",
                position_ids.device.type,
                0,
                name_to_np_type["position_ids"],
                list(position_ids.size()),
                position_ids.data_ptr(),
            )

        # Bind outputs
        for output in ort_session.get_outputs():
            output_name = output.name
            output_buffer = output_buffers[output_name]
            logger.debug(f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}")
            io_binding.bind_output(
                output_name,
                output_buffer.device.type,
                0,
                name_to_np_type[output_name],
                output_shapes[output_name],
                output_buffer.data_ptr(),
            )

        return io_binding

    @staticmethod
    def get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy=True):
        """Copy results to cpu. Returns a list of numpy array."""
        ort_outputs = []
        for output in ort_session.get_outputs():
            output_name = output.name
            buffer = output_buffers[output_name]
            shape = output_shapes[output_name]
            copy_tensor = buffer[0 : numpy.prod(shape)].reshape(shape).clone().detach()
            if return_numpy:
                ort_outputs.append(copy_tensor.cpu().numpy())
            else:
                ort_outputs.append(copy_tensor)
        return ort_outputs


class CudaSession:
    """Inference Session with IO Binding for ONNX Runtime CUDA or TensorRT provider"""

    def __init__(self, ort_session: InferenceSession, device: torch.device, enable_cuda_graph=False):
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

    def allocate_buffers(self, shape_dict: Dict[str, Tuple[int] | List[int]]):
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

    def infer(self, feed_dict: Dict[str, torch.Tensor]):
        """Bind input tensors and run inference"""
        for name, tensor in feed_dict.items():
            assert isinstance(tensor, torch.Tensor) and tensor.is_contiguous()
            if name in self.input_names:
                if self.enable_cuda_graph:
                    assert self.input_tensors[name].nelement() == tensor.nelement()
                    assert self.input_tensors[name].dtype == tensor.dtype
                    assert tensor.device.type == "cuda"
                    # Please install cuda-python package with a version corresponding to CUDA in your machine.
                    from cuda import cudart

                    # Update input tensor inplace since cuda graph requires input and output has fixed memory address.
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

    @staticmethod
    def get_cuda_provider_options(device_id: int, enable_cuda_graph: bool) -> Dict[str, Any]:
        return {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "enable_cuda_graph": enable_cuda_graph,
        }
