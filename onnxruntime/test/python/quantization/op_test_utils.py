
import onnx
import numpy as np
import onnxruntime
from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader


class TestDataFeeds(CalibrationDataReader):
    def __init__(self, data_feeds):
        '''
        parameter data_feeds: list of input feed, each input feed is diction of {input_name: np_array}
        '''
        self.data_feeds = data_feeds
        self.iter_next = iter(self.data_feeds)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.data_feeds)


def check_op_type_count(testcase, model_path, **kwargs):
    model = onnx.load(Path(model_path))
    optype2count = {}
    for op_type in kwargs:
        optype2count[op_type] = 0
    for node in model.graph.node:
        if node.op_type in optype2count:
            optype2count[node.op_type] += 1
    for op_type in kwargs:
        testcase.assertEqual(kwargs[op_type], optype2count[op_type], 'op_type {} count not same'.format(op_type))


def check_model_correctness(testcase, model_path_origin, model_path_to_check, inputs, rtol=1e-2, atol=0.05):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    origin_sess = onnxruntime.InferenceSession(model_path_origin, sess_options=sess_options, providers=["CPUExecutionProvider"])
    origin_results = origin_sess.run([], inputs)
    target_sess = onnxruntime.InferenceSession(model_path_to_check, sess_options=sess_options,providers=["CPUExecutionProvider"])
    target_results = target_sess.run([], inputs)
    for idx, ref_output in enumerate(origin_results):
        output = target_results[idx]
        np.testing.assert_allclose(ref_output, output, rtol=rtol, atol=atol)
