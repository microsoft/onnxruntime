# --------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


import argparse
import logging
import onnx
import tempfile
from pathlib import Path

import onnxruntime
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Model optimizer and shape inferencer, in preparation for quantization.

Model quantization with QDQ format, i.e. inserting QuantizeLinear/DeQuantizeLinear on
the tensor, requires tensor shape information to perform its best. Currently, shape inferencing
works best with optimized model. As a result, it is highly recommended to run quantization
on optimized model with shape information. This is the tool for optimization and shape
inferencing.

Essentially this tool performs the following three (skippable) steps:

1. Symbolic shape inference.
2. Model optimization
3. ONNX shape inference"""
    )

    parser.add_argument("--input", required=True, help="Path to the input model file")
    parser.add_argument("--output", required=True, help="Path to the output model file")
    parser.add_argument(
        "--skip_optimization",
        type=bool,
        default=False,
        help="Skip model optimization step if true. This may result in ONNX shape"
        " inference failure for some models.",
    )
    parser.add_argument(
        "--skip_onnx_shape",
        type=bool,
        default=False,
        help="Skip ONNX shape inference. Symbolic shape inference is most effective"
        " with transformer based models. Skipping all shape inferences may"
        " reduce the effectiveness of quantization, as a tensor with unknown"
        " shape can not be quantized.",
    )
    parser.add_argument(
        "--skip_symbolic_shape",
        type=bool,
        default=False,
        help="Skip symbolic shape inference. Symbolic shape inference is most"
        " effective with transformer based models. Skipping all shape"
        " inferences may reduce the effectiveness of quantization, as a tensor"
        " with unknown shape can not be quantized.",
    )
    parser.add_argument(
        "--auto_merge",
        help="Automatically merge symbolic dims when confliction happens",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--int_max",
        help="maximum value for integer to be treated as boundless for ops like slice",
        type=int,
        default=2**31 - 1,
    )
    parser.add_argument(
        "--guess_output_rank",
        help="guess output rank to be the same as input 0 for unknown ops",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Prints detailed logs of inference, 0: turn off, 1: warnings, 3: detailed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_as_external_data",
        help="Saving an ONNX model to external data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--all_tensors_to_one_file",
        help="Saving all the external data to one file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--external_data_location",
        help="The file location to save the external file",
        default="./",
    )
    parser.add_argument(
        "--external_data_size_threshold",
        help="The size threshold for external data",
        type=int,
        default=1024,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.skip_optimization and args.skip_onnx_shape and args.skip_symbolic_shape:
        logger.error("Skipping all three steps, nothing to be done. Quitting...")
        quit()

    if (not args.skip_optimization) and args.save_as_external_data:
        logger.error("ORT model optimization does not support external data yet!")
        quit()

    logger.info("input model: " + args.input)
    logger.info("output model " + args.output)
    input_model_path = args.input

    with tempfile.TemporaryDirectory(prefix="pre.quant.") as quant_tmp_dir:
        temp_path = Path(quant_tmp_dir)
        model = None

        if not args.skip_symbolic_shape:
            logger.info("Performing symbolic shape inference...")
            model = SymbolicShapeInference.infer_shapes(
                onnx.load(input_model_path),
                args.int_max,
                args.auto_merge,
                args.guess_output_rank,
                args.verbose,
            )

        if not args.skip_optimization:
            # Use ORT optimizers (native code) to optimize model
            if not args.skip_symbolic_shape:
                # Need to save the inferenced model to file so as to run the optimizer
                input_model_path = str(temp_path / "symbolic_shape_inferred.onnx")
                onnx.save(model, input_model_path)
                model = None

            opt_model_path = str(temp_path / "optimized.onnx")
            sess_option = onnxruntime.SessionOptions()
            sess_option.optimized_model_filepath = opt_model_path
            sess_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
            _ = onnxruntime.InferenceSession(input_model_path, sess_option, providers=["CPUExecutionProvider"])

            input_model_path = opt_model_path

        if not args.skip_onnx_shape:
            # ONNX shape inference.
            # According to docs, infer_shapes_path should be used for 2G+ models.
            # If the skip optimization is specified, we could be dealing with a
            # large model. So be on the safe side, save the model
            if model is not None:
                input_model_path = str(temp_path / "symbolic_shape_inferred.onnx")
                if args.save_as_external_data:
                    onnx.save_model(
                        model,
                        input_model_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=args.all_tensors_to_one_file,
                        size_threshold=args.external_data_size_threshold,
                        convert_attribute=False,
                    )
                else:
                    onnx.save(model, input_model_path)
                model = None

            inferred_model_path = str(temp_path / "onnx_shape_inferred.onnx")
            onnx.shape_inference.infer_shapes_path(input_model_path, inferred_model_path)
            model = onnx.load(inferred_model_path)

            # Add inference meta data to onnx model
            metadata_props = {"onnx.infer": "onnxruntime.quant"}
            if model.metadata_props:
                for p in model.metadata_props:
                    metadata_props.update({p.key: p.value})
            onnx.helper.set_model_props(model, metadata_props)

    if model is None:
        model = onnx.load(input_model_path)

    if args.save_as_external_data:
        onnx.save_model(
            model,
            args.output,
            save_as_external_data=True,
            all_tensors_to_one_file=args.all_tensors_to_one_file,
            location=args.external_data_location,
            size_threshold=args.external_data_size_threshold,
            convert_attribute=False,
        )
    else:
        onnx.save(model, args.output)
