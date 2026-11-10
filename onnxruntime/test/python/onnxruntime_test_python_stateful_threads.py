# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Deterministic GIL/cancellation tests using a bounded native-only test barrier."""

import ctypes
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from onnx import TensorProto, helper
from onnxruntime_test_python_stateful import STATEFUL_AVAILABLE, scalar

import onnxruntime as ort
from onnxruntime import stateful as sf


@unittest.skipUnless(STATEFUL_AVAILABLE, "Stateful execution is unavailable in this build")
class TestStatefulThreads(unittest.TestCase):
    def setUp(self):
        name = "stateful_test_barrier.dll" if sys.platform == "win32" else "libstateful_test_barrier.so"
        if sys.platform == "darwin":
            name = "libstateful_test_barrier.dylib"
        path = Path(os.environ.get("ORT_STATEFUL_TEST_BARRIER", str(Path.cwd() / name)))
        self.assertTrue(path.is_file(), f"Build the stateful_test_barrier target: {path}")
        self.library = ctypes.CDLL(str(path))
        for operation in ("Reset", "Blocked", "Release"):
            function = getattr(self.library, f"StatefulBarrier{operation}")
            function.argtypes = []
            function.restype = ctypes.c_int
        self.library.StatefulBarrierWait.argtypes = [ctypes.c_int, ctypes.c_int]
        self.library.StatefulBarrierWait.restype = ctypes.c_int
        self.assertEqual(self.library.StatefulBarrierReset(), 0)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.register_custom_ops_library(str(path))
        graph = helper.make_graph(
            [helper.make_node("StatefulBarrier", ["x"], ["y"], domain="test.stateful")],
            "stateful_gil_barrier",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("test.stateful", 1)],
            ir_version=10,
        )
        self.session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        self.loop = sf.SessionLoop(
            self.session, sf.LoopConfig(("y",), (sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, 1),))
        )

    def tearDown(self):
        self.assertEqual(self.library.StatefulBarrierRelease(), 0)
        self.loop.close()

    def test_native_run_releases_gil_cancel_is_safe_and_busy_operations_reject(self):
        execution = self.loop.create_execution({"x": scalar(3)})
        self._assert_cancellation_and_busy_rejection(execution)

    def test_chain_cancellation_releases_gil_and_requires_reset(self):
        component = sf.ChainComponent(
            "blocked",
            self.loop,
            (sf.InputBinding("x", sf.ValueSource(sf.SourceKind.EXTERNAL_INPUT, "x")),),
        )
        with sf.SessionChain(sf.ChainConfig((component,), (sf.ChainOutput("y", 0, "y"),))) as chain:
            execution = chain.create_execution({"x": scalar(3)})
        self._assert_cancellation_and_busy_rejection(execution)

    def _assert_cancellation_and_busy_rejection(self, execution):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(execution.run)
            try:
                self.assertEqual(self.library.StatefulBarrierWait(1, 2000), 1)
                self.assertEqual(self.library.StatefulBarrierBlocked(), 1)
                self.assertFalse(future.done())
                for operation in (execution.step, execution.run, execution.reset, execution.get_state, execution.close):
                    with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                        operation()
                execution.cancel()
            finally:
                self.assertEqual(self.library.StatefulBarrierRelease(), 0)
            with self.assertRaises(ort.capi.onnxruntime_pybind11_state.Fail):
                future.result(timeout=5)
        self.assertTrue(execution.requires_reset())
        execution.reset()
        self.assertEqual(execution.run().get_outputs()["y"].numpy().item(), 3)
        execution.close()

    def test_independent_contexts_run_concurrently_on_the_same_native_session(self):
        first = self.loop.create_execution({"x": scalar(2)})
        second = self.loop.create_execution({"x": scalar(5)})
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(first.run)
            b = pool.submit(second.run)
            try:
                self.assertEqual(self.library.StatefulBarrierWait(2, 2000), 1)
                self.assertEqual(self.library.StatefulBarrierBlocked(), 2)
                self.assertFalse(a.done())
                self.assertFalse(b.done())
            finally:
                self.assertEqual(self.library.StatefulBarrierRelease(), 0)
            self.assertEqual(a.result(timeout=5).get_outputs()["y"].numpy().item(), 2)
            self.assertEqual(b.result(timeout=5).get_outputs()["y"].numpy().item(), 5)
        first.close()
        second.close()


if __name__ == "__main__":
    unittest.main()
