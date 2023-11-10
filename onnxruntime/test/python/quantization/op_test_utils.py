import uuid
from pathlib import Path

import numpy as np
import onnx

import onnxruntime
from onnxruntime.quantization import CalibrationDataReader


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


def input_feeds_neg_one_zero_one(n, name2shape):
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


def check_model_correctness(testcase, model_path_origin, model_path_to_check, inputs, rtol=1e-2, atol=0.05):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    origin_sess = onnxruntime.InferenceSession(
        model_path_origin, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    origin_results = origin_sess.run([], inputs)
    # enable QDQ transformers
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    target_sess = onnxruntime.InferenceSession(
        model_path_to_check,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    target_results = target_sess.run([], inputs)
    testcase.assertEqual(len(origin_results), len(target_results), "result count are different")
    for idx, ref_output in enumerate(origin_results):
        output = target_results[idx]
        np.testing.assert_allclose(ref_output, output, rtol=rtol, atol=atol)


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
                    testcase.assertTrue(init.data_type == check_item[2])


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
