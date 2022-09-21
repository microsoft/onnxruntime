# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import pathlib
import unittest

from testfixtures import LogCapture

from ..usability_checker import analyze_model

# example usage from <ort root>/tools/python
# python -m unittest util/mobile_helpers/test/test_usability_checker.py
# NOTE: at least on Windows you must use that as the working directory for all the imports to be happy

script_dir = pathlib.Path(__file__).parent
ort_root = script_dir.parents[4]

skip_optimize = False


def _create_logger():
    logger = logging.getLogger("default")
    logger.setLevel(logging.DEBUG)
    return logger


class TestAnalyzer(unittest.TestCase):
    def test_mnist(self):
        """
        Test MNIST which should be fully covered by both NNAPI and CoreML as is.
        :return:
        """
        with LogCapture() as log_capture:
            logger = _create_logger()
            model_path = ort_root / "onnxruntime" / "test" / "testdata" / "mnist.onnx"
            analyze_model(model_path, skip_optimize, logger)

            # print(log_capture)
            log_capture.check_present(
                ("default", "INFO", "1 partitions with a total of 8/8 nodes can be handled by the NNAPI EP."),
                ("default", "INFO", "Model should perform well with NNAPI as is: YES"),
                ("default", "INFO", "1 partitions with a total of 8/8 nodes can be handled by the CoreML EP."),
                ("default", "INFO", "Model should perform well with CoreML as is: YES"),
            )

    def test_scan_model(self):
        """
        Test a Speech model where all the top level nodes are Scan. All the real operators are in subgraphs, so we
        don't use NNAPI/CoreML currently. We want to make sure nodes in subgraphs are counted.
        """
        with LogCapture() as log_capture:
            logger = _create_logger()
            # mnist - should have perfect coverage
            model_path = ort_root / "onnxruntime" / "test" / "testdata" / "scan_1.onnx"
            analyze_model(model_path, skip_optimize, logger)

            # print(log_capture)
            log_capture.check_present(
                ("default", "INFO", "0 partitions with a total of 0/76 nodes can be handled by the NNAPI EP."),
                ("default", "INFO", "72 nodes are in subgraphs, which are currently not handled."),
                ("default", "INFO", "Unsupported ops: ai.onnx:Scan"),
                ("default", "INFO", "Model should perform well with NNAPI as is: NO"),
                ("default", "INFO", "0 partitions with a total of 0/76 nodes can be handled by the CoreML EP."),
                ("default", "INFO", "Model should perform well with CoreML as is: NO"),
            )

    def test_dynamic_shape(self):
        """
        Test a model with dynamic input shape and supported op.
        If we make the shape fixed it should report it will run well with NNAPI/CoreML.
        """
        with LogCapture() as log_capture:
            logger = _create_logger()
            model_path = ort_root / "onnxruntime" / "test" / "testdata" / "abs_free_dimensions.onnx"
            analyze_model(model_path, skip_optimize, logger)

            # print(log_capture)
            log_capture.check_present(
                ("default", "INFO", "0 partitions with a total of 0/1 nodes can be handled by the NNAPI EP."),
                ("default", "INFO", "Model should perform well with NNAPI as is: NO"),
                ("default", "INFO", "Model should perform well with NNAPI if modified to have fixed input shapes: YES"),
                ("default", "INFO", "0 partitions with a total of 0/1 nodes can be handled by the CoreML EP."),
                ("default", "INFO", "CoreML cannot run any nodes in this model."),
                ("default", "INFO", "Model should perform well with CoreML as is: NO"),
                ("default", "INFO", "Model should perform well with CoreML if modified to have fixed input shapes: NO"),
            )

    def test_multi_partitions(self):
        """
        Test a model that breaks into too many partitions to be recommended for use with NNAPI/CoreML
        """
        with LogCapture() as log_capture:
            logger = _create_logger()
            model_path = ort_root / "onnxruntime" / "test" / "testdata" / "gh_issue_9671.onnx"
            analyze_model(model_path, skip_optimize, logger)

            # print(log_capture)
            log_capture.check_present(
                ("default", "INFO", "3 partitions with a total of 17/46 nodes can be handled by the NNAPI EP."),
                ("default", "INFO", "Partition sizes: [5, 4, 8]"),
                (
                    "default",
                    "INFO",
                    "Unsupported ops: ai.onnx:Gather,ai.onnx:ReduceProd,ai.onnx:ReduceSum,"
                    "ai.onnx:Shape,ai.onnx:Unsqueeze",
                ),
                (
                    "default",
                    "INFO",
                    "NNAPI is not recommended with this model as there are 3 partitions "
                    "covering 37.0% of the nodes in the model. "
                    "This will most likely result in worse performance than just using the CPU EP.",
                ),
                ("default", "INFO", "Model should perform well with NNAPI as is: NO"),
                ("default", "INFO", "Partition information if the model was updated to make the shapes fixed:"),
                ("default", "INFO", "3 partitions with a total of 23/46 nodes can be handled by the NNAPI EP."),
                ("default", "INFO", "Partition sizes: [3, 12, 8]"),
                ("default", "INFO", "3 partitions with a total of 15/46 nodes can be handled by the CoreML EP."),
                ("default", "INFO", "Partition sizes: [4, 4, 7]"),
                ("default", "INFO", "Model should perform well with CoreML as is: NO"),
            )
