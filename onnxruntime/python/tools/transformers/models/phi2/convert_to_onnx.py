# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import logging
import onnx
import os
import torch

from dynamo_onnx_helper import DynamoOnnxHelper
from enum import Enum
from onnx import ModelProto, TensorProto, helper
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
from transformers import AutoConfig, AutoModelForCausalLM

## uncomment the following lines to use the local files instead of the pip installed version
## --------------------------------------------------------------------------
# import sys

# sys.path.append(os.path.dirname(__file__))

# transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if transformers_dir not in sys.path:
#     sys.path.append(transformers_dir)
## --------------------------------------------------------------------------

from benchmark_helper import Precision


class AttentionOpType(Enum):
    Attention = "Attention"
    MultiHeadAttention = "MultiHeadAttention"
    GroupQueryAttention = "GroupQueryAttention"

    def __str__(self):
        return self.value


def env_reset():
    for flag in ["AttentionOpType"]:
        if flag in os.environ:
            del os.environ[flag]


class ConvertPhi2ToONNX(DynamoOnnxHelper):
    def __init__(
        self,
        device: torch.device,
        model_class: str = "microsoft/phi-2",
        cache_dir: str = "./cache",
    ):
        self.model_class = model_class
        self.device = device
        self.cache_dir = cache_dir
        self.phi_config = AutoConfig.from_pretrained(self.model_class, trust_remote_code=True, cache_dir=self.cache_dir)
        self.phi_model = None
        self.batch_size = 2
        self.sequence_length = 8
        self.phi2_edge_dict = self.get_phi2_edge_dict(self.phi_config)
        self.attn_op_type = None
        self.precision = None

    def init_attn_type_and_precision(self, attn_op_type: AttentionOpType, precision: Precision):
        self.attn_op_type = attn_op_type
        self.precision = precision

        env_reset()
        os.environ["AttentionOpType"] = str(attn_op_type)

    def get_phi2_edge_dict(self, config: AutoConfig) -> dict:
        edge_dict = {}
        edge_dict["lm_head_1"] = "logits"
        edge_dict["l_input_ids_"] = "input_ids"
        edge_dict["key_states"] = "past_key_0"
        edge_dict["value_states"] = "past_value_0"
        for i in range(config.num_hidden_layers):
            edge_dict[f"key_states_{i}"] = f"past_key_{i}"
            edge_dict[f"value_states_{i}"] = f"past_value_{i}"
            edge_dict[f"model_layers_{i}_1"] = f"present_key_{i}"
            edge_dict[f"model_layers_{i}_1_1"] = f"present_value_{i}"
        return edge_dict

    def simplify_phi2_op_type(self, onnx_model: ModelProto):
        phi2_transformer_layer_name = "modeling_phi_PhiDecoderLayer_model_layers"
        for node in onnx_model.graph.node:
            index = node.op_type.find(phi2_transformer_layer_name)
            if index != -1:
                node.op_type = node.op_type[index:]

        return onnx_model

    def process_graph_io(self, config: AutoConfig, onnx_model: ModelProto):
        use_attn = self.attn_op_type == AttentionOpType.Attention
        graph = onnx_model.graph
        new_inputs = []
        for i, vi in enumerate(graph.input):
            if "input_ids" in vi.name:
                vi = helper.make_tensor_value_info(
                    vi.name,
                    elem_type=TensorProto.INT32,
                    shape=["batch_size", "seq_len"],
                )
                # "Step" is not needed in Attention, we add it here to make the inputs consistent
                vi_pid = helper.make_tensor_value_info(
                    "step",
                    elem_type=TensorProto.INT64,
                    shape=[1],
                )
                vi_mask = helper.make_tensor_value_info(
                    "attention_mask",
                    elem_type=TensorProto.INT32,
                    shape=["batch_size", "seq_len"],
                )
                new_inputs.extend([vi, vi_pid, vi_mask])
            if not use_attn:
                if "past_key" in vi.name or "past_value" in vi.name:
                    vi_cache = helper.make_tensor_value_info(
                        vi.name,
                        elem_type=vi.type.tensor_type.elem_type,
                        shape=[
                            "batch_size",
                            config.num_attention_heads,
                            "past_seq_len",
                            config.hidden_size // config.num_attention_heads,
                        ],
                    )
                    new_inputs.extend([vi_cache])
            else:
                if "past_key" in vi.name:
                    vi_cache = helper.make_tensor_value_info(
                        vi.name.replace("past_key", "past"),
                        elem_type=vi.type.tensor_type.elem_type,
                        shape=[
                            2,
                            "batch_size",
                            config.num_attention_heads,
                            "past_seq_len",
                            config.hidden_size // config.num_attention_heads,
                        ],
                    )
                    new_inputs.extend([vi_cache])

        graph.ClearField("input")
        graph.input.extend(new_inputs)

        new_outputs = []
        for i, vi in enumerate(graph.output):
            if i == 0:
                vi = helper.make_tensor_value_info(
                    vi.name, elem_type=vi.type.tensor_type.elem_type, shape=["batch_size", "seq_len", config.vocab_size]
                )
                new_outputs.extend([vi])
            else:
                if not use_attn:
                    vi = helper.make_tensor_value_info(
                        vi.name,
                        elem_type=vi.type.tensor_type.elem_type,
                        shape=[
                            "batch_size",
                            config.num_attention_heads,
                            "total_seq_len",
                            config.hidden_size // config.num_attention_heads,
                        ],
                    )
                    new_outputs.extend([vi])
                else:
                    if "present_key" in vi.name:
                        vi = helper.make_tensor_value_info(
                            vi.name.replace("present_key", "present"),
                            elem_type=vi.type.tensor_type.elem_type,
                            shape=[
                                2,
                                "batch_size",
                                config.num_attention_heads,
                                "total_seq_len",
                                config.hidden_size // config.num_attention_heads,
                            ],
                        )
                        new_outputs.extend([vi])

        graph.ClearField("output")
        graph.output.extend(new_outputs)

        return onnx_model

    def get_phi2_torch_model(self):
        logging.info("Loading phi2 torch model...")
        if self.phi_model is not None:
            return
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            self.model_class, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.phi_model.eval()
        self.phi_model.to(self.device)

    def get_phi2_torch_inputs(self, batch_size: int, sequence_length: int):
        input_ids = torch.randint(
            low=0,
            high=self.phi_config.vocab_size,
            size=(batch_size, sequence_length),
            dtype=torch.int64,
            device=self.device,
        )
        self.get_phi2_torch_model()
        torch_inputs = self.phi_model.prepare_inputs_for_generation(
            input_ids, past_key_values=self.phi_model(input_ids, use_cache=True)["past_key_values"]
        )
        return torch_inputs["input_ids"], torch_inputs["attention_mask"], torch_inputs["past_key_values"]

    def dynamo_export(self, onnx_path: str):
        input_ids, attention_mask, past_key_values = self.get_phi2_torch_inputs(self.batch_size, self.sequence_length)
        self.phi_model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)

        from torch._dynamo import config

        config.capture_scalar_outputs = True

        logging.info("Exporting Phi2 torch model to ONNX...")
        torch.onnx.dynamo_export(
            self.phi_model,
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
        ).save(onnx_path)
        onnx.checker.check_model(onnx_path)
        onnx.shape_inference.infer_shapes_path(onnx_path)

    def preprocess_onnx(self, onnx_path_in: str, onnx_path_out: str, func_name: str):
        model = onnx.load_model(onnx_path_in, load_external_data=True)
        function_name = None
        for func in model.functions:
            if func.name.endswith(func_name):
                function_name = func.name
                break
        assert function_name is not None
        model = self.unroll_function(model, function_name)
        model = self.update_edges(model, self.phi2_edge_dict)
        model = self.simplify_phi2_op_type(model)
        model = self.remove_dropout_layer(model)
        onnx.save_model(
            model,
            onnx_path_out,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=onnx_path_out + ".data",
        )

    def optimize_phi2_onnx(self, onnx_path: str, onnx_path_opt: str):
        from fusion_options import FusionOptions
        from optimizer import optimize_model
        import uuid

        processed_onnx_path = f"{uuid.uuid1()}.onnx"
        model = onnx.load_model(onnx_path, load_external_data=True)
        model = self.process_graph_io(self.phi_config, model)
        onnx.save_model(
            model,
            processed_onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=processed_onnx_path + ".data",
        )

        optimization_options = FusionOptions("phi")
        optimizer = optimize_model(
            processed_onnx_path,
            model_type="phi",
            num_heads=self.phi_config.num_attention_heads,
            hidden_size=self.phi_config.hidden_size,
            opt_level=0,
            optimization_options=optimization_options,
            only_onnxruntime=False,
        )

        self.erase_onnx_model(processed_onnx_path)

        if self.precision == Precision.FLOAT32:
            optimizer.save_model_to_file(onnx_path_opt, use_external_data_format=True)
            return

        if (
            self.precision == Precision.FLOAT16 or self.precision == Precision.INT4
        ) and self.attn_op_type != AttentionOpType.MultiHeadAttention:
            node_block_list = [
                "GroupQueryAttention_29",
                "GroupQueryAttention_30",
                "GroupQueryAttention_31",
                "Attention_29",
                "Attention_30",
                "Attention_31",
            ]
            logging.info(f"Converting onnx model to float16/bfloat16...")
            optimizer.convert_float_to_float16(
                keep_io_types=False,
                node_block_list=node_block_list,
                use_symbolic_shape_infer=True,
                use_bfloat16_as_blocked_nodes_dtype=True
                if self.attn_op_type == AttentionOpType.GroupQueryAttention
                else False,
            )

        if self.precision == Precision.FLOAT16:
            optimizer.save_model_to_file(onnx_path_opt, use_external_data_format=True)
            return
        else:
            assert self.precision == Precision.INT4
            quant = MatMul4BitsQuantizer(
                model=optimizer.model,
                block_size=16,
                is_symmetric=True,
            )
            quant.process()
            quant.model.save_model_to_file(onnx_path_opt, use_external_data_format=True)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp32_cpu",
        required=False,
        action="store_true",
        help="Generate fp32 ONNX model for CPU",
    )

    parser.add_argument(
        "--int4_cpu",
        required=False,
        action="store_true",
        help="Generate int4 ONNX model for CPU",
    )

    parser.add_argument(
        "--fp32_gpu",
        required=False,
        action="store_true",
        help="Generate fp32 ONNX model for Nvidia GPUs",
    )

    parser.add_argument(
        "--fp16_gpu",
        required=False,
        action="store_true",
        help="Generate fp16 ONNX model for Nvidia GPUs",
    )

    parser.add_argument(
        "--int4_gpu",
        required=False,
        action="store_true",
        help="Generate int4 ONNX model for Nvidia GPUs",
    )

    parser.add_argument(
        "--fp16_a100",
        required=False,
        action="store_true",
        help="Generate fp16 ONNX model for Nvidia A100",
    )

    parser.add_argument(
        "--int4_a100",
        required=False,
        action="store_true",
        help="Generate int4 ONNX model for Nvidia A100",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite existing ONNX models",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default="./cache",
        help="The cache directory for the pytorch model",
    )

    parser.add_argument(
        "--device_id",
        required=False,
        type=int,
        default=0,
        help="The device id for the pytorch model",
    )

    parser.add_argument(
        "--run_example",
        required=False,
        action="store_true",
        help="Run ORT inference example",
    )

    parser.add_argument(
        "--skip_export",
        required=False,
        action="store_true",
        help="Skip exporting ONNX model",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    device = torch.device("cuda", args.device_id) if torch.cuda.is_available() else torch.device("cpu")

    converter = ConvertPhi2ToONNX(device, cache_dir=args.cache_dir)

    temp_onnx_path = "phi2_temp.onnx"
    original_onnx_path = "phi2.onnx"

    if not os.path.exists(original_onnx_path) or args.overwrite:
        converter.dynamo_export(temp_onnx_path)
        converter.preprocess_onnx(
            temp_onnx_path,
            original_onnx_path,
            func_name="modeling_phi_PhiModel_model_1",  # The function to unroll
        )
        converter.erase_onnx_model(temp_onnx_path)

    model_type_to_args = {
        "fp32_cpu": (AttentionOpType.MultiHeadAttention, Precision.FLOAT32, "phi2_decoder_fp32_cpu.onnx"),
        "int4_cpu": (AttentionOpType.MultiHeadAttention, Precision.INT4, "phi2_decoder_int4_cpu.onnx"),
        "fp32_gpu": (AttentionOpType.Attention, Precision.FLOAT32, "phi2_decoder_fp32_gpu.onnx"),
        "fp16_gpu": (AttentionOpType.Attention, Precision.FLOAT16, "phi2_decoder_fp16_gpu.onnx"),
        "int4_gpu": (AttentionOpType.Attention, Precision.INT4, "phi2_decoder_int4_gpu.onnx"),
        "fp16_a100": (AttentionOpType.GroupQueryAttention, Precision.FLOAT16, "phi2_decoder_fp16_a100.onnx"),
        "int4_a100": (AttentionOpType.GroupQueryAttention, Precision.INT4, "phi2_decoder_int4_a100.onnx"),
    }

    if not args.skip_export:
        from multiprocessing import Process

        def run_optimize_phi2_onnx(
            converter: ConvertPhi2ToONNX,
            original_onnx_path: str,
            attention_type: AttentionOpType,
            precision: Precision,
            optimized_onnx_path: str,
        ):
            converter.init_attn_type_and_precision(attention_type, precision)
            converter.optimize_phi2_onnx(original_onnx_path, optimized_onnx_path)

        processes = []
        if args.fp32_cpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["fp32_cpu"])
                )
            )

        if args.int4_cpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["int4_cpu"])
                )
            )

        if args.fp32_gpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["fp32_gpu"])
                )
            )

        if args.fp16_gpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["fp16_gpu"])
                )
            )

        if args.int4_gpu:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx, args=(converter, original_onnx_path, *model_type_to_args["int4_gpu"])
                )
            )

        if args.fp16_a100:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx,
                    args=(converter, original_onnx_path, *model_type_to_args["fp16_a100"]),
                )
            )

        if args.int4_a100:
            processes.append(
                Process(
                    target=run_optimize_phi2_onnx,
                    args=(converter, original_onnx_path, *model_type_to_args["int4_a100"]),
                )
            )

        [p.start() for p in processes]
        [p.join() for p in processes]

    if args.run_example:
        from inference_example import run_phi2

        if args.fp16_a100:
            run_phi2(
                onnx_model_path=model_type_to_args["fp16_a100"][2],
                use_buffer_share=True,
                device_id=args.device_id,
            )
        if args.int4_a100:
            run_phi2(
                onnx_model_path=model_type_to_args["int4_a100"][2],
                use_buffer_share=True,
                device_id=args.device_id,
            )
        if args.fp32_gpu:
            run_phi2(
                onnx_model_path=model_type_to_args["fp32_gpu"][2],
                use_buffer_share=False,
                device_id=args.device_id,
                packed_kv=True,
                use_fp16=False,
            )
        if args.fp16_gpu:
            run_phi2(
                onnx_model_path=model_type_to_args["fp16_gpu"][2],
                use_buffer_share=False,
                device_id=args.device_id,
                packed_kv=True,
            )
        if args.int4_gpu:
            run_phi2(
                onnx_model_path=model_type_to_args["int4_gpu"][2],
                use_buffer_share=False,
                device_id=args.device_id,
                packed_kv=True,
            )
        if args.fp32_cpu or args.int4_cpu:
            raise NotImplementedError("CPU inference example is not implemented yet.")


if __name__ == "__main__":
    main()
