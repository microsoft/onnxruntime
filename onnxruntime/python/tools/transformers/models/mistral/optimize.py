import onnx
from convert_generation import replace_mha_with_gqa
from onnx_model import OnnxModel
from optimizer import optimize_model
from fusion_options import FusionOptions
from transformers import AutoConfig
import os
import argparse
from benchmark_helper import setup_logger
import logging

logger = logging.getLogger("")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--onnx-model-path",
        required=True,
        help="ONNX model to quantize and optimize",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Location for FP16 fused ONNX model",
    )

    parser.add_argument(
        "-hf",
        "--hf-model-path",
        required=True,
        help="Huggingface model for config values, ex. mistralai/Mistral-7B-v0.1",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logs",
    )
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    setup_logger(args.verbose)

    logger.info(f"Optimizing and quantizing model {args.onnx_model_path} to {args.output_path}!")

    config = AutoConfig.from_pretrained(args.hf_model_path)

    tmp_file = args.output_path + ".tmp"

    # Optimize
    optimization_options = FusionOptions("gpt2")

    model_opt = optimize_model(
        args.onnx_model_path,
        model_type="gpt2",
        num_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
        opt_level=0,
        optimization_options=optimization_options,
        only_onnxruntime=False,
    )

    model_opt.save_model_to_file(tmp_file, use_external_data_format=True)
    logger.info(f"The ONNX model at {args.onnx_model_path} has been successfully fused and saved at {tmp_file}!")

    del model_opt
    opt_model = OnnxModel(onnx.load(tmp_file, load_external_data=True))
    opt_model.convert_float_to_float16(keep_io_types=False)
    opt_model = replace_mha_with_gqa(opt_model, "past_sequence_length", config.num_key_value_heads, config.sliding_window)

    logger.info(f"The ONNX model at {tmp_file} has been successfully integrated with GQA and quantized!")

    opt_model.prune_graph()
    opt_model.update_graph(allow_remove_graph_inputs=True)
    opt_model.save_model_to_file(args.output_path, use_external_data_format = True)

    logger.info(f"The ONNX model at {tmp_file} has been successfully pruned and saved at {args.output_path}!")

    data_path = os.path.join(tmp_file + ".data")
    os.remove(tmp_file)
    os.remove(data_path)

    logger.info(f"Temporary file {tmp_file} has been successfully deleted.")

if __name__ == "__main__":
    main()
