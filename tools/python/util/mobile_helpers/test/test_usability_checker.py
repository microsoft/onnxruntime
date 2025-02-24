# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

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
                (
                    "default",
                    "INFO",
                    "1 partitions with a total of 8/8 nodes can be handled by the CoreML NeuralNetwork EP.",
                ),
                ("default", "INFO", "Model should perform well with CoreML NeuralNetwork as is: YES"),
                (
                    "default",
                    "INFO",
                    "1 partitions with a total of 8/8 nodes can be handled by the CoreML MLProgram EP.",
                ),
                ("default", "INFO", "Model should perform well with CoreML MLProgram as is: YES"),
            )

    def test_scan_model(self):
        """
        Test a Speech model where all the top level nodes are Scan. We want to make sure nodes in subgraphs are counted.
        """
        with LogCapture() as log_capture:
            logger = _create_logger()
            # mnist - should have perfect coverage
            model_path = ort_root / "onnxruntime" / "test" / "testdata" / "scan_1.onnx"
            analyze_model(model_path, skip_optimize, logger)

            # print(log_capture)
            log_capture.check_present(
                ("default", "INFO", "4 partitions with a total of 72/76 nodes can be handled by the NNAPI EP."),
                ("default", "INFO", "72 nodes are in 4 subgraphs. Check EP as to whether subgraphs are supported."),
                ("default", "INFO", "Model should perform well with NNAPI as is: NO"),
                (
                    "default",
                    "INFO",
                    "4 partitions with a total of 60/76 nodes can be handled by the CoreML NeuralNetwork EP.",
                ),
                ("default", "INFO", "Model should perform well with CoreML NeuralNetwork as is: NO"),
                (
                    "default",
                    "INFO",
                    "12 partitions with a total of 24/76 nodes can be handled by the CoreML MLProgram EP.",
                ),
                ("default", "INFO", "Model should perform well with CoreML MLProgram as is: NO"),
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
                (
                    "default",
                    "INFO",
                    "0 partitions with a total of 0/1 nodes can be handled by the CoreML MLProgram EP.",
                ),
                ("default", "INFO", "CoreML MLProgram cannot run any nodes in this model."),
                ("default", "INFO", "Model should perform well with CoreML MLProgram as is: NO"),
                (
                    "default",
                    "INFO",
                    "Model should perform well with CoreML MLProgram if modified to have fixed input shapes: NO",
                ),
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
                ("default", "INFO", "3 partitions with a total of 22/50 nodes can be handled by the NNAPI EP."),
                ("default", "INFO", "\tPartition sizes: [13, 2, 7]"),
                (
                    "default",
                    "INFO",
                    "\tUnsupported ops: ai.onnx:ReduceProd,ai.onnx:ReduceSum,ai.onnx:Shape",
                ),
                (
                    "default",
                    "INFO",
                    "NNAPI is not recommended with this model as there are 3 partitions "
                    "covering 44.0% of the nodes in the model. "
                    "This will most likely result in worse performance than just using the CPU EP.",
                ),
                (
                    "default",
                    "INFO",
                    "4 partitions with a total of 20/50 nodes can be handled by the CoreML NeuralNetwork EP.",
                ),
                ("default", "INFO", "\tPartition sizes: [11, 3, 5, 1]"),
            )
