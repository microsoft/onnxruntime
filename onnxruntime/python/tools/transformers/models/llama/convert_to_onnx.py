import argparse
import logging
import os
import tempfile
from itertools import chain
from typing import List

import onnx
import torch
from benchmark_helper import Precision, prepare_environment, setup_logger
from llama_inputs import get_sample_inputs, get_sample_with_past_kv_inputs
from llama_parity import main as parity_check
from onnx_model import OnnxModel
from transformers import LlamaConfig, LlamaForCausalLM

from onnxruntime import quantization as ort_quantization

logger = logging.getLogger("")


def get_model_dynamic_axes(input_names: List[str], output_names: List[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in input_names:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "sequence_length"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes


def get_model_with_past_kv_dynamic_axes(input_names: List[str], output_names: List[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"input_ids", "position_ids"}:
            # shape is (batch_size, 1)
            dynamic_axes[name] = {0: "batch_size"}
        elif name == "attention_mask":
            # shape is (batch_size, past_sequence_length + 1)
            dynamic_axes[name] = {0: "batch_size", 1: "past_sequence_length + 1"}
        elif "past" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, 1, vocab_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + 1, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length + 1"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes


def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False,
    )


# Notes:
# 1) Dynamo export will not work automatically until this issue is resolved: https://github.com/microsoft/onnxscript/issues/493
#
# 2) Dynamo export will run manually if you set the ONNX file path to the same path that you use to save the model after export.
# In other words, the value of `temp_path` should be set as the ONNX file path. You can open the issue in your browser to find
# the location in ONNX Script where you have to make this change.
#
# Once the issue is resolved, we hope to modify the code below as follows for each export.
#
# Before:
# temp_dir = args.output
# temp_path = os.path.join(temp_dir, "temp.onnx")
# ...
# ...
# ...
# del onnx_model
# os.system(f"rm {os.path.join(temp_dir, 'model.*')} && rm {os.path.join(temp_dir, '*.weight')} && rm {temp_path}")
#
#
# After:
# temp_dir = tempfile.TemporaryDirectory()
# temp_path = os.path.join(temp_dir.name, "temp.onnx")
# ...
# ...
# ...
# del onnx_model
# temp_dir.cleanup()
#
def run_dynamo_export(args: argparse.Namespace, l_config: LlamaConfig, llama: LlamaForCausalLM):
    from torch._dynamo import config

    config.capture_scalar_outputs = True

    # Dummy values for export
    batch_size, sequence_length = 2, 8
    device = torch.device("cpu")

    # Export decoder_model.onnx
    input_ids, attn_mask, pos_ids = get_sample_inputs(l_config, device, batch_size, sequence_length)
    temp_dir = args.output  # tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir, "temp.onnx")  # os.path.join(temp_dir.name, "temp.onnx")
    torch.onnx.dynamo_export(
        llama, input_ids, attn_mask, pos_ids, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
    ).save(temp_path)

    # Check decoder_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = os.path.join(args.output, f"{args.model_name}_decoder_model_fp32.onnx")
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(onnx_model, output_path, f"{args.model_name}_decoder_model_fp32.onnx.data")
    del onnx_model
    os.system(
        f"rm {os.path.join(temp_dir, 'model.*')} && rm {os.path.join(temp_dir, '*.weight')} && rm {temp_path}"
    )  # temp_dir.cleanup()

    # Export decoder_with_past_model.onnx
    input_ids, attn_mask, pos_ids, past_kv = get_sample_with_past_kv_inputs(
        l_config, device, batch_size, sequence_length
    )
    temp_dir = args.output  # tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir, "temp.onnx")  # os.path.join(temp_dir.name, "temp.onnx")
    torch.onnx.dynamo_export(
        llama, input_ids, attn_mask, pos_ids, past_kv, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
    ).save(temp_path)

    # Check decoder_with_past_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = os.path.join(args.output, f"{args.model_name}_decoder_with_past_model_fp32.onnx")
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(onnx_model, output_path, f"{args.model_name}_decoder_with_past_model_fp32.onnx.data")
    del onnx_model
    os.system(
        f"rm {os.path.join(temp_dir, 'model.*')} && rm {os.path.join(temp_dir, '*.weight')} && rm {temp_path}"
    )  # temp_dir.cleanup()

    logger.info(f"The {args.model_name} ONNX model has been successfully created with the Dynamo exporter!")


def run_torchscript_export(args: argparse.Namespace, l_config: LlamaConfig, llama: LlamaForCausalLM):
    # Dummy values for export
    batch_size, sequence_length = 2, 8
    device = torch.device("cpu")

    # Export decoder_model.onnx
    decoder_inputs = get_sample_inputs(l_config, device, batch_size, sequence_length)

    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = [
        "logits",
        *list(
            chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(l_config.num_hidden_layers))
        ),
    ]
    dynamic_axes = get_model_dynamic_axes(input_names, output_names)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, "temp.onnx")
    torch.onnx.export(
        llama,
        args=decoder_inputs,
        f=temp_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True,
        verbose=args.verbose,
    )

    # Check decoder_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = os.path.join(args.output, f"{args.model_name}_decoder_model_fp32.onnx")
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(
        onnx_model,
        output_path,
        f"{args.model_name}_decoder_model_fp32.onnx.data",
    )
    del onnx_model
    temp_dir.cleanup()

    # Export decoder_with_past_model.onnx
    decoder_with_past_inputs = get_sample_with_past_kv_inputs(l_config, device, batch_size, sequence_length)
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *list(
            chain.from_iterable(
                (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(l_config.num_hidden_layers)
            )
        ),
    ]
    output_names = [
        "logits",
        *list(
            chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(l_config.num_hidden_layers))
        ),
    ]
    dynamic_axes = get_model_with_past_kv_dynamic_axes(input_names, output_names)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, "temp.onnx")
    torch.onnx.export(
        llama,
        args=decoder_with_past_inputs,
        f=temp_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True,
        verbose=args.verbose,
    )

    # Check decoder_with_past_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = os.path.join(args.output, f"{args.model_name}_decoder_with_past_model_fp32.onnx")
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(
        onnx_model,
        output_path,
        f"{args.model_name}_decoder_with_past_model_fp32.onnx.data",
    )
    del onnx_model
    temp_dir.cleanup()

    logger.info(f"The {args.model_name} ONNX model has been successfully created with the TorchScript exporter!")


def remove_existing_files(output_path: str):
    for filename in os.listdir(output_path):
        filepath = os.path.join(output_path, filename)
        if ".onnx" in filename or ".onnx.data" in filename:
            os.remove(filepath)
            logger.warning(f"Removing {filepath}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help="Model name in Hugging Face",
    )

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default=os.path.join("."),
        help="Directory path to PyTorch model and associated files if saved on disk",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=os.path.join(".", "llama_onnx_models"),
        help="Directory path to save exported model files in",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=[Precision.FLOAT32, Precision.FLOAT16, Precision.INT8],
        help="Precision to export model in",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        required=False,
        default="cpu",
        choices=["cpu", "cuda", "rocm"],
        help="Execution provider to verify parity with",
    )

    parser.add_argument(
        "-q",
        "--quantization_method",
        default="",
        choices=["smooth_quant", "quantize_dynamic"],
        help="Run a specific quantization algorithm. Need to install extra packages in `requirements-quant.txt` for SmoothQuant.",
    )

    smooth_quant_group = parser.add_argument_group("smooth_quant")

    smooth_quant_group.add_argument(
        "--smooth_quant_alpha",
        required=False,
        default=0.8,
        type=float,
        help="Strength to control migration difficulty from activation to weights. Default is 0.8 to match value \
              used in original paper for LLaMA. Paper recommends using values in [0.4, 0.6] range. \
              Link to paper: https://arxiv.org/pdf/2211.10438.pdf",
    )

    smooth_quant_group.add_argument(
        "--smooth_quant_dataset",
        required=False,
        default="NeelNanda/pile-10k",
        help="Path to dataset for calibration during quantization",
    )

    smooth_quant_group.add_argument(
        "--pad_max",
        required=False,
        default=196,
        type=int,
        help="Max padding size",
    )

    smooth_quant_group.add_argument(
        "--calibration_sampling_size",
        required=False,
        type=int,
        default=8,
        help="Calibration sampling size for quantization config",
    )

    smooth_quant_group.add_argument(
        "--nc_workspace",
        required=False,
        type=str,
        default=os.path.join(".", "nc_workspace"),
        help="Workspace to save intermediate files generated by Intel's Neural Compressor package.",
    )

    quantize_dynamic_group = parser.add_argument_group("quantize_dynamic")

    quantize_dynamic_group.add_argument(
        "--quantize_embedding_layer",
        required=False,
        action="store_true",
        help="Quantize MatMul, GEMM, and Gather.",
    )
    quantize_dynamic_group.set_defaults(quantize_embedding_layer=False)

    quantize_dynamic_group.add_argument(
        "--quantize_per_channel",
        required=False,
        action="store_true",
        help="Quantize weights per each channel.",
    )
    quantize_dynamic_group.set_defaults(quantize_per_channel=False)

    quantize_dynamic_group.add_argument(
        "--quantize_reduce_range",
        required=False,
        action="store_true",
        help="Quantize weights with 7 bits.",
    )
    quantize_dynamic_group.set_defaults(quantize_reduce_range=False)

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logs",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "-d",
        "--use_dynamo_export",
        action="store_true",
        help="Use the new Dynamo exporter instead of the old TorchScript exporter",
    )
    parser.set_defaults(use_dynamo_export=False)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    setup_logger(args.verbose)
    prepare_environment(args.input, args.output, args.execution_provider != "cpu")
    remove_existing_files(args.output)
    logger.info(f"Arguments: {args}")

    # Load model and config
    use_auth_token = args.input == os.path.join(".")
    setattr(args, "use_auth_token", use_auth_token)  # noqa: B010
    l_config = LlamaConfig.from_pretrained(
        args.model_name if use_auth_token else args.input, use_auth_token=use_auth_token
    )
    llama = LlamaForCausalLM.from_pretrained(
        args.model_name if use_auth_token else args.input, use_auth_token=use_auth_token, use_cache=True
    )
    original_model_name = args.model_name
    setattr(args, "original_model_name", original_model_name)  # noqa: B010
    args.model_name = args.model_name.split("/")[-1]

    # Export to ONNX
    if args.use_dynamo_export:
        logger.warning("Please ensure you have installed PyTorch, ONNX, and ONNX Script as follows.")
        logger.warning("Step 1 - PyTorch nightly: https://pytorch.org/get-started/locally/")
        logger.warning("Step 2 - ONNX weekly: https://pypi.org/project/onnx-weekly/")
        logger.warning(
            "Step 3 - ONNX Script from source: https://github.com/microsoft/onnxscript#installing-onnx-script"
        )
        logger.warning(
            "Note: After you install ONNX weekly, omit `onnx` when running the first line for installing ONNX Script. This is because you already installed `onnx-weekly` in the previous step."
        )
        run_dynamo_export(args, l_config, llama)
    else:
        run_torchscript_export(args, l_config, llama)

    # Change precision of exported models if not FP32
    decoder_model_fp32_path = os.path.join(args.output, f"{args.model_name}_decoder_model_fp32.onnx")
    decoder_with_past_model_fp32_path = os.path.join(
        args.output, f"{args.model_name}_decoder_with_past_model_fp32.onnx"
    )

    if args.precision == Precision.FLOAT16:
        # Convert decoder_model.onnx to FP16
        decoder_model_fp16_path = os.path.join(args.output, f"{args.model_name}_decoder_model_fp16.onnx")
        model = OnnxModel(onnx.load_model(decoder_model_fp32_path, load_external_data=True))
        model.convert_float_to_float16(keep_io_types=False, op_block_list=["If"])
        model.save_model_to_file(decoder_model_fp16_path, use_external_data_format=True, all_tensors_to_one_file=True)
        del model

        # Convert decoder_with_past_model.onnx to FP16
        decoder_with_past_model_fp16_path = os.path.join(
            args.output, f"{args.model_name}_decoder_with_past_model_fp16.onnx"
        )
        model = OnnxModel(onnx.load_model(decoder_with_past_model_fp32_path, load_external_data=True))
        model.convert_float_to_float16(keep_io_types=False, op_block_list=["If"])
        model.save_model_to_file(
            decoder_with_past_model_fp16_path, use_external_data_format=True, all_tensors_to_one_file=True
        )
        del model

    elif args.precision == Precision.INT8:
        decoder_model_int8_path = os.path.join(args.output, f"{args.model_name}_decoder_model_int8.onnx")
        decoder_with_past_model_int8_path = os.path.join(
            args.output, f"{args.model_name}_decoder_with_past_model_int8.onnx"
        )

        if args.quantization_method == "smooth_quant":
            from neural_compressor import PostTrainingQuantConfig
            from neural_compressor import quantization as intel_quantization
            from neural_compressor import set_workspace
            from onnx.external_data_helper import load_external_data_for_model
            from quant_kv_dataloader import QuantKVDataLoader

            set_workspace(args.nc_workspace)
            quantization_config = PostTrainingQuantConfig(
                calibration_sampling_size=[args.calibration_sampling_size],
                recipes={
                    "optypes_to_exclude_output_quant": ["MatMul"],
                    "smooth_quant": args.smooth_quant,
                    "smooth_quant_args": {"alpha": args.smooth_quant_alpha},
                },
                op_type_dict={
                    "^((?!(MatMul|Gather|Conv)).)*$": {
                        "weight": {"dtype": ["fp32"]},
                        "activation": {"dtype": ["fp32"]},
                    }
                },
            )

            # Convert decoder_model.onnx to INT8
            decoder_model_int8 = intel_quantization.fit(
                decoder_model_fp32_path,
                quantization_config,
                calib_dataloader=QuantKVDataLoader(args),
            )
            load_external_data_for_model(
                decoder_model_int8._model,
                os.path.split(decoder_model_int8._model_path)[0],
            )
            save_onnx_model(
                decoder_model_int8._model,
                decoder_model_int8_path,
                f"{args.model_name}_decoder_model_int8.onnx.data",
            )
            del decoder_model_int8

            # Convert decoder_with_past_model.onnx to INT8
            decoder_with_past_model_int8 = intel_quantization.fit(
                decoder_with_past_model_fp32_path,
                quantization_config,
                calib_dataloader=QuantKVDataLoader(args, onnx_model_path=decoder_model_fp32_path),
            )
            load_external_data_for_model(
                decoder_with_past_model_int8._model,
                os.path.split(decoder_with_past_model_int8._model_path)[0],
            )
            save_onnx_model(
                decoder_with_past_model_int8._model,
                decoder_with_past_model_int8_path,
                f"{args.model_name}_decoder_with_past_model_int8.onnx.data",
            )
            del decoder_with_past_model_int8

            logger.info(f"Removing {args.nc_workspace}")
            os.system(f"rm -R {args.nc_workspace}")

        elif args.quantization_method == "quantize_dynamic":
            logger.warning(
                "The `quantize_dynamic` method is deprecated in favor of `smooth_quant` instead. Precision loss may be high with `quantize_dynamic`."
            )

            # Convert decoder_model.onnx to INT8
            ort_quantization.quantize_dynamic(
                decoder_model_fp32_path,
                decoder_model_int8_path,
                op_types_to_quantize=["MatMul", "Gemm", "Gather"]
                if args.quantize_embedding_layer
                else ["MatMul", "Gemm"],
                per_channel=args.quantize_per_channel,
                reduce_range=args.quantize_reduce_range,
                use_external_data_format=True,
                extra_options={"MatMulConstBOnly": True},
            )

            # Convert decoder_with_past_model.onnx to INT8
            ort_quantization.quantize_dynamic(
                decoder_with_past_model_fp32_path,
                decoder_with_past_model_int8_path,
                op_types_to_quantize=["MatMul", "Gemm", "Gather"]
                if args.quantize_embedding_layer
                else ["MatMul", "Gemm"],
                per_channel=args.quantize_per_channel,
                reduce_range=args.quantize_reduce_range,
                use_external_data_format=True,
                extra_options={"MatMulConstBOnly": True},
            )

        else:
            raise Exception(f"Could not recognize {args.quantization_method} as a quantization method")

    # Verify parity on all saved ONNX models
    del llama  # Delete LLaMA model from memory since it will be loaded again during parity check
    logger.info("Verifying parity on all ONNX models created")
    for filename in os.listdir(args.output):
        if ".data" in filename or ".onnx" not in filename:
            continue

        precision = filename[filename.rfind("_") + 1 : filename.find(".onnx")]
        parity_cmd = ["-m", f"{original_model_name}", "-o", f"{os.path.join(args.output, filename)}", "-fp", precision]
        if "with_past" in filename:
            parity_cmd.append("--use_past_kv")
        parity_check(parity_cmd)


if __name__ == "__main__":
    main()
