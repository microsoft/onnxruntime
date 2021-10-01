#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# Convert Bert ONNX model converted from TensorFlow or exported from PyTorch to use Attention, Gelu,
# SkipLayerNormalization and EmbedLayerNormalization ops to optimize
# performance on NVidia GPU and CPU.
#
# For Bert model exported from PyTorch, OnnxRuntime has bert model optimization support internally.
# You can use the option --use_onnxruntime to check optimizations from OnnxRuntime.
# For Bert model file like name.onnx, optimized model for GPU or CPU from OnnxRuntime will output as
# name_ort_gpu.onnx or name_ort_cpu.onnx in the same directory.
#
# This script is retained for experiment purpose. Useful senarios like the following:
#  (1) Change model from fp32 to fp16 for mixed precision inference in GPU with Tensor Core.
#  (2) Change input data type from int64 to int32.
#  (3) Some model cannot be handled by OnnxRuntime, and you can modify this script to get optimized model.

import logging
import coloredlogs
import os
import argparse
from typing import Dict, Optional
from onnx import load_model, ModelProto
from onnx_model_bart import BartOnnxModel
from onnx_model_bert import BertOnnxModel
from onnx_model_bert_tf import BertOnnxModelTF
from onnx_model_bert_keras import BertOnnxModelKeras
from onnx_model_gpt2 import Gpt2OnnxModel
from fusion_options import FusionOptions

logger = logging.getLogger(__name__)

# Map model type to tuple: optimizer class, export tools (pytorch, tf2onnx, keras2onnx), and default opt_level
MODEL_TYPES = {
    "bart": (BartOnnxModel, "pytorch", 1),
    "bert": (BertOnnxModel, "pytorch", 1),
    "bert_tf": (BertOnnxModelTF, "tf2onnx", 0),
    "bert_keras": (BertOnnxModelKeras, "keras2onnx", 0),
    "gpt2": (Gpt2OnnxModel, "pytorch", 1),
    "gpt2_tf": (Gpt2OnnxModel, 'tf2onnx', 0)  # might add a class for GPT2OnnxModel for TF later.
}


def optimize_by_onnxruntime(onnx_model_path: str,
                            use_gpu: bool = False,
                            optimized_model_path: Optional[str] = None,
                            opt_level: Optional[int] = 99,
                            disabled_optimizers=[]) -> str:
    """
    Use onnxruntime to optimize model.

    Args:
        onnx_model_path (str): the path of input onnx model.
        use_gpu (bool): whether the optimized model is targeted to run in GPU.
        optimized_model_path (str or None): the path of optimized model.
        opt_level (int): graph optimization level.
        disabled_optimizers (List[str]): a list of names of disabled optimizers
    Returns:
        optimized_model_path (str): the path of optimized model
    """
    assert opt_level in [1, 2, 99]
    import onnxruntime

    if use_gpu and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        logger.error("There is no gpu for onnxruntime to do optimization.")
        return onnx_model_path

    sess_options = onnxruntime.SessionOptions()
    if opt_level == 1:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == 2:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if optimized_model_path is None:
        path_prefix = onnx_model_path[:-5]  #remove .onnx suffix
        optimized_model_path = "{}_o{}_{}.onnx".format(path_prefix, opt_level, "gpu" if use_gpu else "cpu")

    sess_options.optimized_model_filepath = optimized_model_path

    kwargs = {}
    if disabled_optimizers:
        kwargs["disabled_optimizers"] = disabled_optimizers

    if not use_gpu:
        session = onnxruntime.InferenceSession(onnx_model_path,
                                               sess_options,
                                               providers=['CPUExecutionProvider'],
                                               **kwargs)
    else:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options, **kwargs)
        assert 'CUDAExecutionProvider' in session.get_providers()  # Make sure there is GPU

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    logger.debug("Save optimized model by onnxruntime to {}".format(optimized_model_path))
    return optimized_model_path


def optimize_by_fusion(model: ModelProto,
                       model_type: str = 'bert',
                       num_heads: int = 0,
                       hidden_size: int = 0,
                       optimization_options: Optional[FusionOptions] = None):
    """ Optimize Model by graph fusion logic.

    Note that ONNXRuntime graph optimizations (like constant folding) will not be applied. So it is better to enable
    constant folding during exporting ONNX model, or run optimize_by_onnxruntime on the model first like optimize_model.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need specify these parameters.

    Args:
        model (ModelProto): model object
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically (for model_type "bert" only). 
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically (for model_type "bert" only). 
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions. Defaults to None.

     Returns:
        object of an optimizer class.
    """
    if model_type != "bert" and (num_heads == 0 or hidden_size == 0):
        logger.warning("Please specify parameters of num_heads and hidden_size when model_type is not 'bert'")

    (optimizer_class, producer, _) = MODEL_TYPES[model_type]

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f"Model producer not matched: Expect {producer}, Got {model.producer_name} {model.producer_version}. Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = FusionOptions(model_type)

    optimizer = optimizer_class(model, num_heads, hidden_size)

    optimizer.optimize(optimization_options)

    optimizer.topological_sort()

    optimizer.model.producer_name = "onnxruntime.transformers"
    from onnxruntime import __version__ as onnxruntime_version
    optimizer.model.producer_version = onnxruntime_version

    return optimizer


def optimize_model(input: str,
                   model_type: str = 'bert',
                   num_heads: int = 0,
                   hidden_size: int = 0,
                   optimization_options: Optional[FusionOptions] = None,
                   opt_level: int = None,
                   use_gpu: bool = False,
                   only_onnxruntime: bool = False):
    """ Optimize Model by OnnxRuntime and/or python fusion logic.

    ONNX Runtime has graph optimizations (https://onnxruntime.ai/docs/resources/graph-optimizations.html). 
    However, the coverage is limited. We also have graph fusions that implemented in Python to improve the coverage.
    They can combined: ONNX Runtime will run first when opt_level > 0, then graph fusions in Python will be applied.

    To use ONNX Runtime only and no Python fusion logic, use only_onnxruntime flag and a positive opt_level like
        optimize_model(input, opt_level=1, use_gpu=False, only_onnxruntime=True)

    When opt_level is None, we will choose default optimization level according to model type.

    When opt_level is 0 and only_onnxruntime is False, only python fusion logic is used and onnxruntime is disabled.

    When opt_level > 1, use_gpu shall set properly since the optimized graph might contain operators for GPU or CPU only. 
    If your model is intended for GPU inference only (especially float16 or mixed precision model), it is recommended to 
    set use_gpu to be True, otherwise the model is not optimized for GPU inference.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need specify these parameters.

    Args:
        input (str): input model path.
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically (for model_type "bert" only). 
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically (for model_type "bert" only). 
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions. Defaults to None.
        opt_level (int, optional): onnxruntime graph optimization level (0, 1, 2 or 99) or None. Defaults to None.
                                   When the value is None, default value (1 for bert and gpt2, 0 for other model types) will be used.
                                   When the level > 0, onnxruntime will be used to optimize model first.
        use_gpu (bool, optional): use gpu or not for onnxruntime. Defaults to False.
        only_onnxruntime (bool, optional): only use onnxruntime to optimize model, and no python fusion. Defaults to False.

     Returns:
        object of an optimizer class.
    """
    assert opt_level is None or opt_level in [0, 1, 2, 99]

    if model_type != "bert" and (num_heads == 0 or hidden_size == 0):
        logger.warning("Please specify parameters of num_heads and hidden_size when model_type is not 'bert'")

    (optimizer_class, producer, default_opt_level) = MODEL_TYPES[model_type]

    if opt_level is None:
        opt_level = default_opt_level

    temp_model_path = None
    if opt_level > 1:
        # Disable some optimizers that might cause failure in symbolic shape inference or attention fusion.
        disabled_optimizers = [] if only_onnxruntime else [
            'MatMulScaleFusion', 'MatMulAddFusion'
            'SimplifiedLayerNormFusion', 'GemmActivationFusion', 'BiasSoftmaxFusion'
        ]
        temp_model_path = optimize_by_onnxruntime(input,
                                                  use_gpu=use_gpu,
                                                  opt_level=opt_level,
                                                  disabled_optimizers=disabled_optimizers)
    elif opt_level == 1:
        # basic optimizations (like constant folding and cast elimation) are not specified to exection provider.
        # CPU provider is used here so that there is no extra node for GPU memory copy.
        temp_model_path = optimize_by_onnxruntime(input, use_gpu=False, opt_level=1)

    if only_onnxruntime and not temp_model_path:
        logger.warning("Please specify a positive value for opt_level when only_onnxruntime is True")

    model = load_model(temp_model_path or input)

    if only_onnxruntime:
        optimizer = optimizer_class(model, num_heads, hidden_size)
    else:
        optimizer = optimize_by_fusion(model, model_type, num_heads, hidden_size, optimization_options)

    # Remove the temporary model.
    if temp_model_path:
        os.remove(temp_model_path)
        logger.debug("Remove tempoary model: {}".format(temp_model_path))

    return optimizer


def get_fusion_statistics(optimized_model_path: str) -> Dict[str, int]:
    """
    Get counter of fused operators in optimized model.

    Args:
        optimized_model_path (str): the path of onnx model.

    Returns:
        A dictionary with operator type as key, and count as value
    """
    model = load_model(optimized_model_path, format=None, load_external_data=True)
    optimizer = BertOnnxModel(model)
    return optimizer.get_fused_operator_statistics()


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=
        'Graph optimization tool for ONNX Runtime. It transforms ONNX graph to use optimized operators for Transformer models.'
    )
    parser.add_argument('--input', required=True, type=str, help="input onnx model path")

    parser.add_argument('--output', required=True, type=str, help="optimized onnx model path")

    parser.add_argument('--model_type',
                        required=False,
                        type=str.lower,
                        default="bert",
                        choices=list(MODEL_TYPES.keys()),
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPES.keys()))

    parser.add_argument(
        '--num_heads',
        required=False,
        type=int,
        default=0,
        help=
        "number of attention heads like 12 for bert-base and 16 for bert-large. Default is 0 to detect automatically for BERT. For other model type, this parameter need specify correctly."
    )

    parser.add_argument(
        '--hidden_size',
        required=False,
        type=int,
        default=0,
        help=
        "hidden size like 768 for bert-base and 1024 for bert-large. Default is 0 to detect automatically for BERT. For other model type, this parameter need specify correctly."
    )

    parser.add_argument(
        '--input_int32',
        required=False,
        action='store_true',
        help=
        "Use int32 (instead of int64) inputs. It could avoid unnecessary data cast when EmbedLayerNormalization is fused for BERT."
    )
    parser.set_defaults(input_int32=False)

    parser.add_argument(
        '--float16',
        required=False,
        action='store_true',
        help=
        "Convert all weights and nodes in float32 to float16. It has potential loss in precision compared to mixed precision conversion (see convert_float_to_float16)."
    )
    parser.set_defaults(float16=False)

    FusionOptions.add_arguments(parser)

    parser.add_argument('--verbose', required=False, action='store_true', help="show debug information.")
    parser.set_defaults(verbose=False)

    parser.add_argument(
        '--use_gpu',
        required=False,
        action='store_true',
        help="Use GPU for inference. Set this flag if your model is intended for GPU when opt_level > 1.")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--only_onnxruntime',
                        required=False,
                        action='store_true',
                        help="optimized by onnxruntime only, and no graph fusion in Python")
    parser.set_defaults(only_onnxruntime=False)

    parser.add_argument(
        '--opt_level',
        required=False,
        type=int,
        choices=[0, 1, 2, 99],
        default=None,
        help=
        "onnxruntime optimization level. 0 will disable onnxruntime graph optimization. The recommended value is 1. When opt_level > 1 is used, optimized model for GPU might not run in CPU. Level 2 and 99 are intended for --only_onnxruntime."
    )

    parser.add_argument('--use_external_data_format',
                        required=False,
                        action='store_true',
                        help="use external data format to store large model (>2GB)")
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()

    return args


def _setup_logger(verbose):
    if verbose:
        coloredlogs.install(level='DEBUG', fmt='[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')
    else:
        coloredlogs.install(fmt='%(funcName)20s: %(message)s')


def main():
    args = _parse_arguments()

    _setup_logger(args.verbose)

    logger.debug(f"arguments:{args}")

    if os.path.realpath(args.input) == os.path.realpath(args.output):
        logger.warning(f"Specified the same input and output path. Note that this may overwrite the original model")

    optimization_options = FusionOptions.parse(args)

    optimizer = optimize_model(args.input,
                               args.model_type,
                               args.num_heads,
                               args.hidden_size,
                               opt_level=args.opt_level,
                               optimization_options=optimization_options,
                               use_gpu=args.use_gpu,
                               only_onnxruntime=args.only_onnxruntime)

    if args.float16:
        optimizer.convert_float_to_float16(keep_io_types=True)

    if args.input_int32:
        optimizer.change_graph_inputs_to_int32()

    optimizer.save_model_to_file(args.output, args.use_external_data_format)

    if optimizer.is_fully_optimized():
        logger.info("The model has been fully optimized.")
    else:
        logger.info("The model has been optimized.")


if __name__ == "__main__":
    main()
