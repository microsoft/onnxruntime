# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import time
from onnxruntime import InferenceSession, SessionOptions, OrtValue
from onnxruntime.transformers.io_binding_helper import TypeHelper
from benchmark import measure_gpu_memory
import numpy as np
import onnxruntime as ort

class OrtModelBinding:
    def __init__(self, ort_session, io_shape, device_id=0):
        for input in ort_session.get_inputs():
            assert input.name in io_shape
        # for output in ort_session.get_outputs():
        #     assert output.name in io_shape
        input_names = [input.name for input in ort_session.get_inputs()]
        output_names = [output.name for output in ort_session.get_outputs()]

        self.io_shape = io_shape
        self.io_numpy_type = TypeHelper.get_io_numpy_type_map(ort_session)
        self.io_binding = ort_session.io_binding()
        self.io_ort_value = {}

        for name in input_names:
            ort_value = OrtValue.ortvalue_from_shape_and_type(
                io_shape[name], self.io_numpy_type[name], "cuda", device_id
            )
            self.io_ort_value[name] = ort_value
            self.io_binding.bind_ortvalue_input(name, ort_value)

        for name in output_names:
            if name in io_shape:
                ort_value = OrtValue.ortvalue_from_shape_and_type(
                    io_shape[name], self.io_numpy_type[name], "cuda", device_id
                )
                self.io_ort_value[name] = ort_value
                self.io_binding.bind_ortvalue_output(name, ort_value)
            else:
                self.io_binding.bind_output(name, 'cuda')

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        type=str,
        default='unet.onnx',
        help="path of unet onnx model.",
    )


    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 10, 16, 32],
        help="Number of images per batch. Default is 1.",
    )

    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt. Default is 1",
    )

    parser.add_argument(
        "--height",
        required=False,
        type=int,
        default=512,
        help="Output image height. Default is 512.",
    )

    parser.add_argument(
        "--width",
        required=False,
        type=int,
        default=512,
        help="Output image width. Default is 512.",
    )

    parser.add_argument(
        "-s",
        "--steps",
        required=False,
        type=int,
        default=5,
        help="Number of steps. Default is 5.",
    )

    args = parser.parse_args()
    return args

def test(args, enable_cuda_graph, output_shape = None):
    print(args)

    start_memory = measure_gpu_memory(None)
    print("GPU memory used before loading models:", start_memory)

    # ort_memory_info = ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
    # ort_arena_cfg = ort.OrtArenaCfg(
    #     {
    #         "max_mem": 4000000000,
    #         "arena_extend_strategy": 1,
    #         "initial_chunk_size_bytes": 2000000000,
    #         "max_dead_bytes_per_chunk": 4,
    #         "initial_growth_chunk_size_bytes": 1000000000,
    #     }
    # )
    # ort.create_and_register_allocator(ort_memory_info, ort_arena_cfg)
    # ort.create_and_register_allocator(ort_memory_info, None)

    options = SessionOptions()
    options.log_severity_level = 0
    # options.add_session_config_entry("session.use_env_allocators", "1")

    load_start = time.time()
    session = InferenceSession(args.input, options, providers=[("CUDAExecutionProvider", {'enable_cuda_graph': enable_cuda_graph}), "CPUExecutionProvider"])
    load_end = time.time()
    print(f"Session creation took {load_end - load_start} seconds")

    # Create dummy inputs
    batch_size = args.batch_size * args.num_images_per_prompt
    channels = 4
    height = args.height // 8
    width = args.width // 8
    sequence_length = 77
    hidden_size = 768  # TODO: parse from graph input shape

    sample = np.random.normal(0, 0.01, size=(2 * batch_size, channels, height, width)).astype('float16')
    # np.ones((2 * batch_size, channels, height, width), dtype=np.float16)
    timestep = np.ones((1), dtype=np.float16)
    encoder_hidden_states = np.random.normal(0, 0.01, size=(2 * batch_size, sequence_length, hidden_size)).astype('float16')
    #np.ones((2 * batch_size, sequence_length, hidden_size), dtype=np.float16)

    io_shape = {
        "sample": list(sample.shape),
        "timestep": list(timestep.shape),
        "encoder_hidden_states": list(encoder_hidden_states.shape),
         #"out_sample": [2 * batch_size, 4, height, width],
    }

    if output_shape:
        io_shape["out_sample"] = output_shape

    model_bindings = OrtModelBinding(session, io_shape)

    def unet_inference():
        for _ in range(args.steps):
            model_bindings.io_ort_value["encoder_hidden_states"].update_inplace(encoder_hidden_states)
            model_bindings.io_ort_value["sample"].update_inplace(sample)
            model_bindings.io_ort_value["timestep"].update_inplace(timestep)
            session.run_with_iobinding(model_bindings.io_binding)
            if output_shape:
                output = model_bindings.io_ort_value["out_sample"].numpy()
            else:
                output = model_bindings.io_binding.get_outputs()[0].numpy()

    start = time.time()
    first_run_memory = measure_gpu_memory(unet_inference, start_memory)
    end = time.time()
    print(f"First inference took {end - start} seconds. Memory usage: {first_run_memory}")

    start = time.time()
    second_run_memory = measure_gpu_memory(unet_inference, start_memory)
    end = time.time()
    print(f"Second inference took {end - start} seconds. Memory usage: {first_run_memory}")

    if output_shape:
        return model_bindings.io_ort_value["out_sample"].numpy()
    else:
        return model_bindings.io_binding.get_outputs()[0].numpy()

def main():
    args = parse_arguments()
    print(args)

    print("**Test without CUDA Graph**")
    output = test(args, False)
    #print("output", output)
    print("output shape", output.shape)

    print("**Test with CUDA Graph**")
    output = test(args, True, output.shape)
    #print("output", output)
    print("output shape", output.shape)

if __name__ == "__main__":
    main()
