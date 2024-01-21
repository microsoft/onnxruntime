#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from argparse import ArgumentParser

import onnx
import tensorrt as trt
from onnx import TensorProto, helper


class TensorRTEngineWrapperCreator:
    def __init__(self, args):
        ctx_embed_mode = args.embed_mode
        engine_cache_path = args.trt_engine_cache_path
        self.model_name = args.model_name
        self.dynamic_dim_count = 0
        self.plugins = args.plugins

        # Get serialized engine from engine cache
        with open(engine_cache_path, "rb") as file:
            engine_buffer = file.read()

        if ctx_embed_mode:
            ep_cache_context_content = engine_buffer
        else:
            ep_cache_context_content = engine_cache_path

        logger = trt.Logger(trt.Logger.WARNING)

        # Enable TRT plugins
        trt.init_libnvinfer_plugins(logger, "")
        if len(self.plugins):
            import ctypes

            ctypes.CDLL(self.plugins)

        # Deserialize an TRT engine
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_buffer)
        num_bindings = engine.num_bindings

        input_tensors = []
        output_tensors = []
        input_tensor_shapes = []
        output_tensor_shapes = []
        input_tensor_types = []
        output_tensor_types = []

        # Get type and shape of each input/output
        for b_index in range(num_bindings):
            tensor_name = engine.get_tensor_name(b_index)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_type = engine.get_tensor_dtype(tensor_name)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_tensors.append(tensor_name)
                input_tensor_shapes.append(tensor_shape)
                input_tensor_types.append(tensor_type)
            else:
                output_tensors.append(tensor_name)
                output_tensor_shapes.append(tensor_shape)
                output_tensor_types.append(tensor_type)

        # Note:
        # The TRT engine should be built with min, max and opt profiles so that dynamic shape input can have dimension of "-1"
        print(input_tensors)
        print(input_tensor_types)
        print(input_tensor_shapes)
        print(output_tensors)
        print(output_tensor_types)
        print(output_tensor_shapes)

        nodes = [
            helper.make_node(
                "EPContext",
                input_tensors,
                output_tensors,
                "EPContext",
                domain="com.microsoft",
                embed_mode=ctx_embed_mode,
                ep_cache_context=ep_cache_context_content,
            ),
        ]

        model_inputs = []
        for i in range(len(input_tensors)):
            model_inputs.append(
                helper.make_tensor_value_info(
                    input_tensors[i],
                    self.trt_data_type_to_onnx_data_type(input_tensor_types[i]),
                    self.trt_shape_to_ort_shape(input_tensor_shapes[i]),
                )
            )

        model_outputs = []
        for i in range(len(output_tensors)):
            model_outputs.append(
                helper.make_tensor_value_info(
                    output_tensors[i],
                    self.trt_data_type_to_onnx_data_type(output_tensor_types[i]),
                    self.trt_shape_to_ort_shape(output_tensor_shapes[i]),
                )
            )

        self.graph = helper.make_graph(
            nodes,
            "trt_engine_wrapper",
            model_inputs,
            model_outputs,
        )

    def trt_data_type_to_onnx_data_type(self, trt_data_type):
        if trt_data_type == trt.DataType.FLOAT:
            return TensorProto.FLOAT
        elif trt_data_type == trt.DataType.HALF:
            return TensorProto.FLOAT16
        elif trt_data_type == trt.DataType.INT8:
            return TensorProto.INT8
        elif trt_data_type == trt.DataType.INT32:
            return TensorProto.INT32
        elif trt_data_type == trt.DataType.BOOL:
            return TensorProto.BOOL
        elif trt_data_type == trt.DataType.UINT8:
            return TensorProto.UINT8
        else:
            return TensorProto.UNDEFINED

    # TRT uses "-1" to represent dynamic dimension
    # ORT uses symbolic name to represent dynamic dimension
    # Here we only do the conversion when there is any dynamic dimension in the shape
    def trt_shape_to_ort_shape(self, trt_data_shape):
        def has_dynamic_dim(trt_data_shape):
            if any(dim == -1 for dim in trt_data_shape):
                return True
            return False

        if not has_dynamic_dim(trt_data_shape):
            return trt_data_shape

        ort_data_shape = []
        if has_dynamic_dim(trt_data_shape):
            for dim in trt_data_shape:
                if dim == -1:
                    ort_data_shape.append("free_dim_" + str(self.dynamic_dim_count))
                    self.dynamic_dim_count += 1
                else:
                    ort_data_shape.append(dim)
        return ort_data_shape

    def create_model(self):
        model = helper.make_model(self.graph)
        onnx.save(model, self.model_name)
        print(self.model_name + " is created.")


def main():
    parser = ArgumentParser("Generate Onnx model which includes the TensorRT engine binary.")
    parser.add_argument(
        "-p", "--trt_engine_cache_path", help="Required. Path to TensorRT engine cache.", required=True, type=str
    )
    parser.add_argument(
        "-e",
        "--embed_mode",
        help="mode 0 means the engine cache path and mode 1 means engine binary data",
        required=False,
        default=0,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        help="Model name to be created",
        required=False,
        default="trt_engine_wrapper.onnx",
        type=str,
    )
    parser.add_argument(
        "--plugins",
        help="List of plugin paths to load",
        required=False,
        default=[],
        nargs="+",
        type=str,
    )
    args = parser.parse_args()
    ctor = TensorRTEngineWrapperCreator(args)
    ctor.create_model()


if __name__ == "__main__":
    main()
