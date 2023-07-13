import uuid
from pathlib import Path

import numpy as np
import onnx
import packaging.version as pv
from onnx import TensorProto
from onnx.numpy_helper import float8e4m3_to_float32
from onnx.helper import float32_to_float8e4m3, np_dtype_to_tensor_dtype
from onnx.reference import ReferenceEvaluator
from onnx.reference.custom_element_types import float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz
from onnx.reference.op_run import OpRun

import onnxruntime
from onnxruntime.quantization import CalibrationDataReader


class QGemm(OpRun):
    op_domain = "com.microsoft"

    f8_types = {
        TensorProto.FLOAT8E4M3FN,
        TensorProto.FLOAT8E4M3FNUZ,
        TensorProto.FLOAT8E5M2,
        TensorProto.FLOAT8E5M2FNUZ,
    }

    def get_zero_point_type(self, zero_point: np.ndarray) -> int:
        if zero_point.dtype == float8e4m3fn and zero_point.dtype.descr[0][0] == "e4m3fn":
            return TensorProto.FLOAT8E4M3FN
        if zero_point.dtype == float8e4m3fnuz and zero_point.dtype.descr[0][0] == "e4m3fnuz":
            return TensorProto.FLOAT8E4M3FNUZ
        if zero_point.dtype == float8e5m2 and zero_point.dtype.descr[0][0] == "e5m2":
            return TensorProto.FLOAT8E5M2
        if zero_point.dtype == float8e5m2fnuz and zero_point.dtype.descr[0][0] == "e5m2fnuz":
            return TensorProto.FLOAT8E5M2FNUZ
        return np_dtype_to_tensor_dtype(zero_point.dtype)

    def _run(
        self,
        A,
        a_scale,
        a_zero_point,
        B,
        b_scale,
        b_zero_point,
        C=None,
        y_scale=None,
        y_zero_point=None,
        transA=0,
        transB=0,
        alpha=1.0,
    ):
        if transA:
            A = A.T
        if transB:
            B = B.T

        a_type = self.get_zero_point_type(a_zero_point)
        b_type = self.get_zero_point_type(b_zero_point)
        y_type = self.get_zero_point_type(y_zero_point)
        if (
            a_type == TensorProto.FLOAT8E4M3FN
            and b_type == TensorProto.FLOAT8E4M3FN
            and y_type == TensorProto.FLOAT8E4M3FN
        ):
            a_scaled = (float8e4m3_to_float32(A).astype(float) - float8e4m3_to_float32(a_zero_point)) * np.float32(
                a_scale
            )
            b_scaled = (float8e4m3_to_float32(B).astype(float) - float8e4m3_to_float32(b_zero_point)) * np.float32(
                b_scale
            )
            y = a_scaled @ b_scaled * np.float32(alpha)
            if C is not None:
                y += C * np.float32(a_scale) * np.float32(b_scale)
            if y_scale is not None:
                y /= y_scale
            if y_zero_point is not None:
                y += float8e4m3_to_float32(y_zero_point)
                ry = y.ravel()

                fy = np.empty(ry.shape, dtype=float8e4m3fn)
                for i in range(fy.shape[0]):
                    el = float32_to_float8e4m3(ry[i])  # type: ignore[assignment]
                    fy[i] = el
                y = fy.reshape(y.shape)
            return (y,)
        elif a_type in self.f8_types or b_type in self.f8_types or y_type in self.f8_types:
            raise NotImplementedError(f"QGemm not implemented for zero_types {a_type}, {b_type}, {y_type}.")
        else:
            a_scaled = (A.astype(float) - a_zero_point) * np.float32(a_scale)
            b_scaled = (B.astype(float) - b_zero_point) * np.float32(b_scale)
            y = a_scaled @ b_scaled * np.float32(alpha)
            if C is not None:
                y += C * np.float32(a_scale) * np.float32(b_scale)
            if y_scale is not None:
                y /= np.float32(y_scale)
            if y_zero_point is not None:
                y += y_zero_point

            if y_zero_point is not None:
                dtype = y_zero_point.dtype
            elif C is not None:
                dtype = C.dtype
            else:
                dtype = A.dtype

            y = np.rint(y)
            if dtype == np.uint8:
                y = np.clip(y, 0, 255)
            elif dtype == np.int8:
                y = np.clip(y, -128, 127)
            else:
                raise ValueError(f"Unexpected dtype={dtype}, it should be uint8 or int8.")

            return (y.astype(dtype),)


class TestDataFeeds(CalibrationDataReader):
    def __init__(self, data_feeds):
        """
        parameter data_feeds: list of input feed, each input feed is diction of {input_name: np_array}
        """
        self.data_feeds = data_feeds
        self.iter_next = iter(self.data_feeds)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.data_feeds)


def InputFeedsNegOneZeroOne(n, name2shape):  # noqa: N802
    """
    randomize n feed according to shape, its values are from -1, 0, and 1
    """
    input_data_list = []
    for _i in range(n):
        inputs = {}
        for name, shape in name2shape.items():
            inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
        input_data_list.extend([inputs])
    dr = TestDataFeeds(input_data_list)
    return dr


def check_op_type_order(testcase, model_to_check, ops):
    if isinstance(model_to_check, str):
        model = onnx.load(model_to_check)
    elif isinstance(model_to_check, onnx.ModelProto):
        model = model_to_check

    testcase.assertEqual(len(ops), len(model.graph.node), "op count is not same")
    for node_idx, node in enumerate(model.graph.node):
        testcase.assertEqual(
            ops[node_idx],
            node.op_type,
            f"op {node_idx} is not in order. Expected: {ops[node_idx]}, Actual: {node.op_type}",
        )


def check_op_type_count(testcase, model_path, **kwargs):
    model = onnx.load(Path(model_path))
    optype2count = {}
    for op_type in kwargs:
        optype2count[op_type] = 0
    for node in model.graph.node:
        if node.op_type in optype2count:
            optype2count[node.op_type] += 1
    for op_type in kwargs:
        testcase.assertEqual(
            kwargs[op_type],
            optype2count[op_type],
            f"op_type {op_type} count not same",
        )


def check_model_correctness(
    testcase, model_path_origin, model_path_to_check, inputs, rtol=1e-2, atol=0.05, providers=None
):
    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.optimized_model_filepath = model_path_to_check + ".optimized.onnx"
    origin_sess = onnxruntime.InferenceSession(model_path_origin, sess_options=sess_options, providers=providers)
    origin_results = origin_sess.run([], inputs)

    if pv.Version(onnx.__version__) >= pv.Version("1.16.0"):
        ref = ReferenceEvaluator(model_path_to_check, new_ops=[QGemm])
        target_results = ref.run(None, inputs)
        testcase.assertEqual(len(origin_results), len(target_results), "result count are different")
        for idx, ref_output in enumerate(origin_results):
            output = target_results[idx]
            np.testing.assert_allclose(
                ref_output,
                output,
                rtol=rtol,
                atol=atol,
                err_msg=f"Model {model_path_to_check!r} failed for providers={providers!r}.",
            )

    # enable QDQ transformers
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    target_sess = onnxruntime.InferenceSession(
        model_path_to_check,
        sess_options=sess_options,
        providers=providers,
    )
    target_results = target_sess.run([], inputs)
    testcase.assertEqual(len(origin_results), len(target_results), "result count are different")
    for idx, ref_output in enumerate(origin_results):
        output = target_results[idx]
        np.testing.assert_allclose(
            ref_output,
            output,
            rtol=rtol,
            atol=atol,
            err_msg=f"Model {model_path_to_check!r} failed for providers={providers!r}.",
        )


def check_op_nodes(testcase, model_path, node_checker):
    model = onnx.load(Path(model_path))
    for node in model.graph.node:
        testcase.assertTrue(node_checker(node))


def check_qtype_by_node_type(testcase, model_to_check, check_list):
    if isinstance(model_to_check, str):
        model = onnx.load(model_to_check)
    elif isinstance(model_to_check, onnx.ModelProto):
        model = model_to_check
    model = onnx.shape_inference.infer_shapes(model)
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    value_infos.update({ot.name: ot for ot in model.graph.output})
    value_infos.update({it.name: it for it in model.graph.input})
    initializers = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type in check_list:
            input_output_check_list = check_list[node.op_type]
            for check_item in input_output_check_list:
                tensor_name = node.input[check_item[1]] if check_item[0] == "i" else node.output[check_item[1]]
                testcase.assertTrue((tensor_name in value_infos) or (tensor_name in initializers))
                if tensor_name in value_infos:
                    vi = value_infos[tensor_name]
                    testcase.assertTrue(vi.type.HasField("tensor_type"))
                    testcase.assertTrue(vi.type.tensor_type.elem_type == check_item[2])
                else:  # if (tensor_name in initializers):
                    init = initializers[tensor_name]
                    testcase.assertEqual(init.data_type, check_item[2])


def create_clip_node(input_name, output_name, node_name, initializers, min_value=-1.0, max_value=1.0):
    clip_min_name = str(uuid.uuid4())
    clip_max_name = str(uuid.uuid4())
    clip_inputs = [input_name, clip_min_name, clip_max_name]
    clip_outputs = [output_name]
    clip_name = node_name
    initializers.append(onnx.numpy_helper.from_array(np.array(min_value, dtype=np.float32), name=clip_min_name))
    initializers.append(onnx.numpy_helper.from_array(np.array(max_value, dtype=np.float32), name=clip_max_name))
    return onnx.helper.make_node("Clip", clip_inputs, clip_outputs, name=clip_name)


def generate_random_initializer(initializer_name, tensor_shape, tensor_dtype, mean=0.0, dev=0.3):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(mean, dev, tensor_shape).astype(tensor_dtype)
    init = onnx.numpy_helper.from_array(tensor, initializer_name)
    return init
