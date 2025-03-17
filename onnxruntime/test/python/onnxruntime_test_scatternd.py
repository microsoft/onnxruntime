import itertools
import json
import os
import typing
import unittest
import warnings

import numpy as np
import onnx.helper as oh
from onnx import TensorProto, load
from onnx.numpy_helper import from_array
from onnx.reference import ReferenceEvaluator

import onnxruntime


def has_cuda():
    available_providers = [provider for provider in onnxruntime.get_available_providers()]
    return "CUDAExecutionProvider" in available_providers


def ignore_warnings(warns: typing.List[Warning]) -> typing.Callable:
    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


class TestScatterPerProvider(unittest.TestCase):
    def assert_exists(self, filename: str):
        assert os.path.exists(filename), f"Unable to find {filename!r}."

    def common_scatter(self, opset, providers, dtype, reduction, expected_names):
        from onnxruntime import InferenceSession, SessionOptions

        op_type = "ScatterElements" if "ScatterElements" in expected_names else "ScatterND"
        ndim = 2 if op_type == "ScatterElements" else 3

        assert dtype in (np.float16, np.float32)
        itype = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("CastLike", ["X", "I"], ["data"]),
                    oh.make_node(
                        op_type,
                        inputs=["data", "indices", "updates"],
                        outputs=["sy"],
                        # axis=0,
                        reduction=reduction,
                    ),
                    oh.make_node("Sub", ["sy", "I"], ["Y"]),
                ],
                "name",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [None] * ndim),
                    oh.make_tensor_value_info("indices", TensorProto.INT64, [None, None]),
                    oh.make_tensor_value_info("updates", itype, [None] * ndim),
                ],
                [oh.make_tensor_value_info("Y", itype, [None] * ndim)],
                [from_array(np.array([0], dtype=dtype), name="I")],
            ),
            opset_imports=[oh.make_opsetid("", opset)],
            ir_version=8 if opset <= 18 else 9,
        )

        if not os.path.exists("temp_dump"):
            os.mkdir("temp_dump")
        for name in os.listdir("temp_dump"):
            os.remove(os.path.join("temp_dump", name))

        filename = f"temp_dump/{op_type}_{providers[0]}_{itype}.onnx"
        opts = SessionOptions()
        opts.optimized_model_filepath = filename
        sess = InferenceSession(model.SerializeToString(), opts, providers=providers)
        self.assertTrue(sess is not None)
        self.assert_exists(filename)
        onx = load(filename)
        names = [n.op_type for n in onx.graph.node]
        self.assertEqual(expected_names, names)

        sonx = str(onx).replace(" ", "").replace("\n", "|")
        sexp = 'op_type:"Cast"|attribute{|name:"to"|type:INT|i:%d|}' % itype  # noqa: UP031
        sexp2 = 'op_type:"Cast"|attribute{|name:"to"|i:%d|type:INT|}' % itype  # noqa: UP031
        assert sexp in sonx or sexp2 in sonx, f"Unable to find a substring in {sonx!r}"
        if providers == ["CPUExecutionProvider"]:
            return

        if op_type == "ScatterElements":
            data = np.zeros((3, 3), dtype=np.float32)
            data[0, 0] = 1
            indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
            updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=dtype)
        else:
            data = np.array(
                [
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                ],
                dtype=np.float32,
            )
            indices = np.array([[0], [2]], dtype=np.int64)
            updates = np.array(
                [
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                ],
                dtype=dtype,
            )
        opts = SessionOptions()
        opts.enable_profiling = True
        opts.optimized_model_filepath = filename
        sess = InferenceSession(model.SerializeToString(), opts, providers=providers)
        got = sess.run(None, {"X": data, "indices": indices, "updates": updates})[0]
        self.assertEqual(got.dtype, updates.dtype)
        prof = sess.end_profiling()

        with open(prof, "r") as f:  # noqa: UP015
            content = f.read()
        js = json.loads(content)

        exe_providers = []
        suffixes = ["_kernel_time", "_fence_before", "_fence_after"]
        rows = []
        for row in js:
            if "args" in row and isinstance(row["args"], dict):
                for k, v in row["args"].items():
                    row[f"args_{k}"] = v
                del row["args"]
            name = row["name"]
            for suf in suffixes:
                if name.endswith(suf):
                    changed = name[: -len(suf)]
                    row["op_name"] = changed
                    break
            rows.append(row)
            exe_providers.append((row.get("args_provider", None), row.get("args_op_name", None)))
        short_list = [(a, b) for a, b in exe_providers if a is not None and b is not None]
        self.assertEqual(short_list, [("CUDAExecutionProvider", o) for o in expected_names])

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    @ignore_warnings(DeprecationWarning)
    def test_scatterels_cuda(self):
        default_value = [
            "Cast",
            "ScatterElements",
            "Sub",
        ]
        expected = {
            (np.float32, "none"): default_value,
            (np.float16, "none"): default_value,
            (np.float32, "add"): default_value,
            (np.float16, "add"): default_value,
            (np.float32, "mul"): default_value,
            (np.float16, "mul"): default_value,
            (np.float32, "min"): default_value,
            (np.float16, "min"): default_value,
            (np.float32, "max"): default_value,
            (np.float16, "max"): default_value,
        }
        for opset, dtype, reduction in itertools.product(
            [16, 18], [np.float32, np.float16], ["none", "add", "mul", "min", "max"]
        ):
            with self.subTest(dtype=dtype, reduction=reduction, opset=opset):
                self.common_scatter(
                    opset,
                    ["CUDAExecutionProvider"],
                    dtype,
                    reduction,
                    expected[dtype, reduction],
                )

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    @ignore_warnings(DeprecationWarning)
    def test_scatternd_cuda(self):
        default_value = [
            "Cast",
            "ScatterND",
            "Sub",
        ]
        expected = {
            (np.float32, "none"): default_value,
            (np.float16, "none"): default_value,
            (np.float32, "add"): default_value,
            (np.float16, "add"): default_value,
            (np.float32, "mul"): default_value,
            (np.float16, "mul"): default_value,
            (np.float32, "min"): default_value,
            (np.float16, "min"): default_value,
            (np.float32, "max"): default_value,
            (np.float16, "max"): default_value,
        }
        for opset, dtype, reduction in itertools.product(
            [16, 18], [np.float32, np.float16], ["none", "add", "mul", "min", "max"]
        ):
            with self.subTest(dtype=dtype, reduction=reduction, opset=opset):
                self.common_scatter(
                    opset,
                    ["CUDAExecutionProvider"],
                    dtype,
                    reduction,
                    expected[dtype, reduction],
                )

    @ignore_warnings(DeprecationWarning)
    def test_scatterels_cpu(self):
        default_value = [
            "Cast",
            "ScatterElements",
            "Sub",
        ]
        expected = {
            (np.float32, "none"): default_value,
            (np.float16, "none"): default_value,
            (np.float32, "add"): default_value,
            (np.float16, "add"): default_value,
            (np.float32, "mul"): default_value,
            (np.float16, "mul"): default_value,
            (np.float32, "min"): default_value,
            (np.float16, "min"): default_value,
            (np.float32, "max"): default_value,
            (np.float16, "max"): default_value,
        }
        for opset, dtype, reduction in itertools.product([16, 18], [np.float32], ["none", "add", "mul", "min", "max"]):
            with self.subTest(dtype=dtype, reduction=reduction, opset=opset):
                self.common_scatter(
                    opset,
                    ["CPUExecutionProvider"],
                    dtype,
                    reduction,
                    expected[dtype, reduction],
                )

    @ignore_warnings(DeprecationWarning)
    def test_scatternd_cpu(self):
        default_value = [
            "Cast",
            "ScatterND",
            "Sub",
        ]
        expected = {
            (np.float32, "none"): default_value,
            (np.float16, "none"): default_value,
            (np.float32, "add"): default_value,
            (np.float16, "add"): default_value,
            (np.float32, "mul"): default_value,
            (np.float16, "mul"): default_value,
            (np.float32, "min"): default_value,
            (np.float16, "min"): default_value,
            (np.float32, "max"): default_value,
            (np.float16, "max"): default_value,
        }
        for opset, dtype, reduction in itertools.product([16, 18], [np.float32], ["none", "add", "mul", "min", "max"]):
            with self.subTest(dtype=dtype, reduction=reduction, opset=opset):
                self.common_scatter(
                    opset,
                    ["CPUExecutionProvider"],
                    dtype,
                    reduction,
                    expected[dtype, reduction],
                )

    def _scatternd_standalone_cuda(self, reduction, line):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ScatterND",
                        inputs=["data", "indices", "updates"],
                        outputs=["y"],
                        reduction=reduction,
                    )
                ],
                "nd",
                [
                    oh.make_tensor_value_info("data", TensorProto.FLOAT, [None, None, None]),
                    oh.make_tensor_value_info("indices", TensorProto.INT64, [None, None]),
                    oh.make_tensor_value_info("updates", TensorProto.FLOAT, [None, None, None]),
                ],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        data = np.full((2, 2, 3), 0.1, dtype=np.float32)
        indices = np.array([[line], [1 - line], [line]], dtype=np.int64)
        updates = (2 ** (np.arange(18) + 1).astype(np.float32).reshape((3, 2, 3))).astype(np.float32)

        feeds = dict(data=data, indices=indices, updates=updates)
        ref = ReferenceEvaluator(model)
        expected = ref.run(None, feeds)[0]

        providers = (
            [
                ["CUDAExecutionProvider"],
                ["CPUExecutionProvider"],
            ]
            if has_cuda()
            else [["CPUExecutionProvider"]]
        )
        for provider in providers:
            sess = onnxruntime.InferenceSession(model.SerializeToString(), providers=provider)
            got = sess.run(None, feeds)[0]
            self.assertEqual(expected.tolist(), got.tolist())

    def test_scatternd_standalone_cuda(self):
        self._scatternd_standalone_cuda("add", 0)
        self._scatternd_standalone_cuda("add", 1)
        self._scatternd_standalone_cuda("mul", 0)
        self._scatternd_standalone_cuda("mul", 1)
        self._scatternd_standalone_cuda("min", 0)
        self._scatternd_standalone_cuda("min", 1)
        self._scatternd_standalone_cuda("max", 0)
        self._scatternd_standalone_cuda("max", 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
