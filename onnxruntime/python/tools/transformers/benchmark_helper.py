# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import sys
import numpy
import time
import argparse
import logging
import torch
import onnx
from enum import Enum

logger = logging.getLogger('')


class Precision(Enum):
    FLOAT32 = 'fp32'
    FLOAT16 = 'fp16'
    INT8 = 'int8'

    def __str__(self):
        return self.value


def create_onnxruntime_session(onnx_model_path, use_gpu, enable_all_optimization=True, num_threads=0, verbose=False):
    session = None
    try:
        from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
        sess_options = SessionOptions()

        if enable_all_optimization:
            sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC

        if num_threads > 0:
            sess_options.intra_op_num_threads = num_threads
            logger.debug(f"Session option: intra_op_num_threads={sess_options.intra_op_num_threads}")

        if verbose:
            sess_options.log_severity_level = 0

        logger.debug(f"Create session for onnx model: {onnx_model_path}")
        execution_providers = ['CPUExecutionProvider'
                               ] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = InferenceSession(onnx_model_path, sess_options, providers=execution_providers)
    except:
        logger.error(f"Exception", exc_info=True)

    return session


def setup_logger(verbose=True):
    # output logging to stdout
    log_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        log_handler.setFormatter(logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s'))
        logging_level = logging.DEBUG
    else:
        log_handler.setFormatter(logging.Formatter('%(filename)20s: %(message)s'))
        logging_level = logging.INFO
        logging.getLogger("transformers").setLevel(logging.ERROR)
    log_handler.setLevel(logging_level)

    # Avoid duplicated handlers when runing this script in multiple cells of Jupyter Notebook.
    if not logger.hasHandlers():
        logger.addHandler(log_handler)

    logger.setLevel(logging_level)


def prepare_environment(cache_dir, output_dir, use_gpu):
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import onnxruntime
    if use_gpu:
        assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(
        ), "Please install onnxruntime-gpu package to test GPU inference."

    import transformers
    logger.info(f'PyTorch Version:{torch.__version__}')
    logger.info(f'Transformers Version:{transformers.__version__}')
    logger.info(f'Onnxruntime Version:{onnxruntime.__version__}')

    from packaging import version
    assert version.parse(torch.__version__) >= version.parse('1.4.0')
    assert version.parse(transformers.__version__) >= version.parse('2.11.0')
    assert version.parse(onnxruntime.__version__) >= version.parse('1.4.0')
