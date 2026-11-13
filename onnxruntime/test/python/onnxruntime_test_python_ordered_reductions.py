# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Stream, CUDA capture, and concurrent-run contracts for opt-in ordered reductions."""

import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

import onnxruntime as ort

try:
    import torch
except ImportError:
    torch = None


def make_model(matrix, width=128):
    shape = ["batch", 128, 128] if matrix else ["batch", width]
    output_shape = ["batch", 128] if matrix else ["batch", 1]
    op = "OrderedMatrixReduce128" if matrix else "OrderedRowReduce"
    attributes = {} if matrix else {"width": width, "normalize": 1}
    node = helper.make_node(op, ["X"], ["Y"], domain="com.microsoft", **attributes)
    graph = helper.make_graph(
        [node],
        "ordered_reduction",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)], ir_version=9
    ).SerializeToString()


@unittest.skipUnless("CUDAExecutionProvider" in ort.get_available_providers(), "CUDA EP is required")
class TestOrderedReductions(unittest.TestCase):
    def test_concurrent_distinct_inputs_and_cuda_placement(self):
        rng = np.random.default_rng(7832)
        with tempfile.TemporaryDirectory() as directory:
            for matrix in (False, True):
                options = ort.SessionOptions()
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(directory) / "ordered")
                session = ort.InferenceSession(make_model(matrix), options, providers=["CUDAExecutionProvider"])
                shape = (30, 128, 128) if matrix else (30, 128)
                inputs = [rng.standard_normal(shape, dtype=np.float32) for _ in range(4)]
                expected = [session.run(None, {"X": value})[0] for value in inputs]
                self.assertFalse(np.array_equal(expected[0].view(np.uint32), expected[1].view(np.uint32)))
                for _ in range(3):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        actual = list(
                            executor.map(lambda value, current=session: current.run(None, {"X": value})[0], inputs)
                        )
                    for result, reference in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(result.view(np.uint32), reference.view(np.uint32))
                events = json.loads(Path(session.end_profiling()).read_text())
                nodes = [event["args"] for event in events if event.get("cat") == "Node"]
                self.assertTrue(nodes)
                self.assertTrue(all(node.get("provider") == "CUDAExecutionProvider" for node in nodes))

    def test_nondefault_stream_external_capture_and_replay(self):
        if torch is None:
            self.skipTest("PyTorch is required to create and capture a nondefault CUDA stream")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is required")
        stream = torch.cuda.Stream()
        self.assertNotEqual(stream.cuda_stream, torch.cuda.default_stream().cuda_stream)
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        run_options = ort.RunOptions()
        run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
        for matrix in (False, True):
            session = ort.InferenceSession(
                make_model(matrix, width=4096),
                options,
                providers=[("CUDAExecutionProvider", {"user_compute_stream": str(stream.cuda_stream)})],
            )
            input_shape = (30, 128, 128) if matrix else (30, 4096)
            output_shape = (30, 128) if matrix else (30, 1)
            with torch.cuda.stream(stream):
                source = torch.empty(input_shape, dtype=torch.float32, device="cuda")
                output = torch.empty(output_shape, dtype=torch.float32, device="cuda")
                binding = session.io_binding()
                binding.bind_input("X", "cuda", 0, np.float32, input_shape, source.data_ptr())
                binding.bind_output("Y", "cuda", 0, np.float32, output_shape, output.data_ptr())
                source.fill_(1)
                for _ in range(3):
                    session.run_with_iobinding(binding, run_options)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            # External capture rejects internal waits and proves launches use this stream.
            with torch.cuda.graph(graph, stream=stream):
                session.run_with_iobinding(binding, run_options)
            for value in (1.0, -2.0, 3.0):
                with torch.cuda.stream(stream):
                    source.fill_(value)
                    graph.replay()
                stream.synchronize()
                expected = value * 128 if matrix else value
                np.testing.assert_array_equal(
                    output.cpu().numpy().view(np.uint32),
                    np.full(output_shape, expected, np.float32).view(np.uint32),
                )


if __name__ == "__main__":
    unittest.main()
