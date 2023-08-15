# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os
import shutil
from collections import OrderedDict
from typing import Any, Dict

import torch

import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import TypeHelper

logger = logging.getLogger(__name__)


class OrtCudaSession:
    """Inference Session with IO Binding for ONNX Runtime CUDA or TensorRT provider"""

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


class Engine(OrtCudaSession):
    def __init__(self, engine_path, provider: str, device_id: int = 0, enable_cuda_graph=False):
        self.engine_path = engine_path
        self.provider = provider
        self.provider_options = self.get_cuda_provider_options(device_id, enable_cuda_graph)

        device = torch.device("cuda", device_id)
        ort_session = ort.InferenceSession(
            self.engine_path,
            providers=[
                (provider, self.provider_options),
                "CPUExecutionProvider",
            ],
        )

        super().__init__(ort_session, device, enable_cuda_graph)

    def get_cuda_provider_options(self, device_id: int, enable_cuda_graph: bool) -> Dict[str, Any]:
        return {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "enable_cuda_graph": enable_cuda_graph,
        }


class Engines:
    def __init__(self, provider, onnx_opset: int = 14):
        self.provider = provider
        self.engines = {}
        self.onnx_opset = onnx_opset

    @staticmethod
    def get_onnx_path(onnx_dir, model_name):
        return os.path.join(onnx_dir, model_name + ".onnx")

    @staticmethod
    def get_engine_path(engine_dir, model_name, profile_id):
        return os.path.join(engine_dir, model_name + profile_id + ".onnx")

    def build(
        self,
        models,
        engine_dir: str,
        onnx_dir: str,
        force_engine_rebuild: bool = False,
        fp16: bool = True,
        device_id: int = 0,
        enable_cuda_graph: bool = False,
    ):
        profile_id = "_fp16" if fp16 else "_fp32"

        if force_engine_rebuild:
            if os.path.isdir(onnx_dir):
                logger.info("Remove existing directory %s since force_engine_rebuild is enabled", onnx_dir)
                shutil.rmtree(onnx_dir)
            if os.path.isdir(engine_dir):
                logger.info("Remove existing directory %s since force_engine_rebuild is enabled", engine_dir)
                shutil.rmtree(engine_dir)

        if not os.path.isdir(engine_dir):
            os.makedirs(engine_dir)

        if not os.path.isdir(onnx_dir):
            os.makedirs(onnx_dir)

        # Export models to ONNX
        for model_name, model_obj in models.items():
            onnx_path = Engines.get_onnx_path(onnx_dir, model_name)
            onnx_opt_path = Engines.get_engine_path(engine_dir, model_name, profile_id)
            if os.path.exists(onnx_opt_path):
                logger.info("Found cached optimized model: %s", onnx_opt_path)
            else:
                if os.path.exists(onnx_path):
                    logger.info("Found cached model: %s", onnx_path)
                else:
                    logger.info("Exporting model: %s", onnx_path)
                    model = model_obj.get_model().to(model_obj.device)
                    with torch.inference_mode():
                        inputs = model_obj.get_sample_input(1, 512, 512)
                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=self.onnx_opset,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

                # Optimize onnx
                logger.info("Generating optimized model: %s", onnx_opt_path)
                model_obj.optimize_ort(onnx_path, onnx_opt_path, to_fp16=fp16)

        for model_name in models:
            engine_path = Engines.get_engine_path(engine_dir, model_name, profile_id)
            engine = Engine(engine_path, self.provider, device_id=device_id, enable_cuda_graph=enable_cuda_graph)
            logger.info("%s options for %s: %s", self.provider, model_name, engine.provider_options)
            self.engines[model_name] = engine

    def get_engine(self, model_name):
        return self.engines[model_name]
