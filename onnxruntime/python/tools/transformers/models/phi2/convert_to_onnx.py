# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import onnx
import torch

from enum import Enum
from onnx import ModelProto, TensorProto, helper
from transformers import AutoConfig, AutoModelForCausalLM
from typing import List

# --------------------------------------------------------------------------
# The following code is used when this file is not in the ORT package
import sys, os

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)
# --------------------------------------------------------------------------

from benchmark_helper import Precision


class AttentionOpType(Enum):
    Attention = "attention"
    MultiHeadAttention = "mha"
    GroupQueryAttention = "gqa"

    def __str__(self):
        return self.value


class ConvertPhi2ToONNX:
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
        self.phi2_edge_dict = self.__get_phi2_edge_dict(self.phi_config)

    def init_attn_type_and_precision(self, attn_op_type: AttentionOpType, precision: Precision):
        self.attn_op_type = attn_op_type
        self.precision = precision

    def __get_phi2_edge_dict(self, config: AutoConfig) -> dict:
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

    def __simplify_phi2_op_type(self, onnx_model: ModelProto):
        phi2_transformer_layer_name = "modeling_phi_PhiDecoderLayer_model_layers"
        for node in onnx_model.graph.node:
            index = node.op_type.find(phi2_transformer_layer_name)
            if index != -1:
                node.op_type = node.op_type[index:]

        return onnx_model

    def __process_graph_io(self, config: AutoConfig, onnx_model: ModelProto):
        use_gqa = self.attn_op_type == AttentionOpType.GroupQueryAttention
        graph = onnx_model.graph
        new_inputs = []
        for i, vi in enumerate(graph.input):
            if "input_ids" in vi.name:
                vi = helper.make_tensor_value_info(
                    vi.name,
                    elem_type=TensorProto.INT32,
                    shape=["batch_size", "seq_len"],
                )
                vi_pid = helper.make_tensor_value_info(
                    "step",
                    elem_type=TensorProto.INT64,
                    shape=[1],
                )
                vi_mask = helper.make_tensor_value_info(
                    "attention_mask",
                    elem_type=TensorProto.INT64 if use_gqa else TensorProto.INT32,
                    shape=["batch_size", "seq_len"],
                )
                new_inputs.extend([vi, vi_pid, vi_mask])
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

        graph.ClearField("input")
        graph.input.extend(new_inputs)

        new_outputs = []
        for i, vi in enumerate(graph.output):
            if i == 0:
                vi = helper.make_tensor_value_info(
                    vi.name, elem_type=vi.type.tensor_type.elem_type, shape=["batch_size", "seq_len", config.vocab_size]
                )
            else:
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

        graph.ClearField("output")
        graph.output.extend(new_outputs)

        return onnx_model

    def __update_edges(self, model: onnx.ModelProto, edge_mapping: dict):
        """
        Updates the edges in the model according to the given mapping.
        """
        for node in model.graph.node:
            for i in range(len(node.input)):
                if node.input[i] in edge_mapping:
                    node.input[i] = edge_mapping[node.input[i]]
            for i in range(len(node.output)):
                if node.output[i] in edge_mapping:
                    node.output[i] = edge_mapping[node.output[i]]

        for graph_input in model.graph.input:
            if graph_input.name in edge_mapping:
                graph_input.name = edge_mapping[graph_input.name]
        for graph_output in model.graph.output:
            if graph_output.name in edge_mapping:
                graph_output.name = edge_mapping[graph_output.name]

        return model

    def __unroll_function(self, model: onnx.ModelProto, func_name: str) -> onnx.ModelProto:
        """
        Unrolls the function with the given name in the model.
        """
        nodes_to_remove = []
        nodes_to_add = []
        edges_to_remove = []
        edges_to_add = []
        for node in model.graph.node:
            if node.op_type == func_name:
                nodes_to_remove.append(node)
                for edge in node.input:
                    edges_to_remove.append(edge)
                for edge in node.output:
                    edges_to_remove.append(edge)

        for f in model.functions:
            if f.name == func_name:
                for node in f.node:
                    nodes_to_add.append(node)
                for edge in f.input:
                    edges_to_add.append(edge)
                for edge in f.output:
                    edges_to_add.append(edge)

        assert len(edges_to_remove) == len(edges_to_add)

        for node in nodes_to_remove:
            model.graph.node.remove(node)
        for node in nodes_to_add:
            model.graph.node.append(node)

        edge_mapping = {}
        for i in range(len(edges_to_remove)):
            k = edges_to_remove[i]
            v = edges_to_add[i]
            if k != v:
                edge_mapping[k] = v

        return self.__update_edges(model, edge_mapping)

    def __remove_dropout_layer(self, model: onnx.ModelProto):
        """
        Removes the dropout layer in the model.
        """
        edge_mapping = {}
        nodes_to_remove = []
        for node in model.graph.node:
            if node.op_type.find("Dropout") != -1:
                assert len(node.input) == 1
                assert len(node.output) == 1
                edge_mapping[node.output[0]] = node.input[0]
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            model.graph.node.remove(node)

        return self.__update_edges(model, edge_mapping)

    def __get_phi2_torch_model(self):
        if self.phi_model is not None:
            return
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            self.model_class, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.phi_model.eval()
        self.phi_model.to(self.device)

    def get_phi2_torch_inputs(self, batch_size: int, sequence_length: int):
        input_ids = torch.randint(
            low=0, high=self.phi_config.vocab_size, size=(batch_size, sequence_length), dtype=torch.int64, device=device
        )
        self.__get_phi2_torch_model()
        torch_inputs = self.phi_model.prepare_inputs_for_generation(
            input_ids, past_key_values=self.phi_model(input_ids, use_cache=True)["past_key_values"]
        )
        return torch_inputs["input_ids"], torch_inputs["attention_mask"], torch_inputs["past_key_values"]

    def erase_onnx_model(self, onnx_path: str):
        assert onnx_path.endswith(".onnx")
        if not os.path.exists(onnx_path):
            return
        model = onnx.load_model(onnx_path, load_external_data=False)
        onnx_data_path = None
        for initializer in model.graph.initializer:
            if initializer.data_location == 1 and initializer.external_data[0].key == "location":
                onnx_data_path = "./" + initializer.external_data[0].value
                break
        os.remove(onnx_path)
        if onnx_data_path is not None:
            os.remove(onnx_data_path)

    def dynamo_export(self, onnx_path: str):
        input_ids, attention_mask, past_key_values = self.get_phi2_torch_inputs(self.batch_size, self.sequence_length)
        self.phi_model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)

        from torch._dynamo import config

        config.capture_scalar_outputs = True
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
        model = self.__unroll_function(model, function_name)
        model = self.__update_edges(model, self.phi2_edge_dict)
        model = self.__simplify_phi2_op_type(model)
        model = self.__process_graph_io(self.phi_config, model)
        model = self.__remove_dropout_layer(model)
        onnx.save_model(
            model,
            onnx_path_out,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=onnx_path_out + ".data",
        )

    def optimize_phi2_onnx(self, onnx_path: str, onnx_path_opt: str, use_fp16: bool = False):
        from fusion_options import FusionOptions
        from optimizer import optimize_model

        optimization_options = FusionOptions("phi")
        optimizer = optimize_model(
            onnx_path,
            model_type="phi",
            num_heads=self.phi_config.num_attention_heads,
            hidden_size=self.phi_config.hidden_size,
            opt_level=0,
            optimization_options=optimization_options,
            only_onnxruntime=False,
        )

        if use_fp16:
            node_block_list = ["GroupQueryAttention_0_29", "GroupQueryAttention_0_30", "GroupQueryAttention_0_31"]
            optimizer.convert_float_to_float16(
                keep_io_types=False,
                node_block_list=node_block_list,
                use_symbolic_shape_infer=True,
                use_bfloat16_as_blocked_nodes_dtype=True,
            )

        optimizer.save_model_to_file(onnx_path_opt, use_external_data_format=True)
        optimizer.get_operator_statistics()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp32_cpu",
        required=False,
        action="store_true",
        help="Generate fp32 onnx model for CPU",
    )

    parser.add_argument(
        "--int4_cpu",
        required=False,
        action="store_true",
        help="Generate int4 onnx model for CPU",
    )

    parser.add_argument(
        "--fp32_gpu",
        required=False,
        action="store_true",
        help="Generate fp32 onnx model for Nvidia GPUs",
    )

    parser.add_argument(
        "--fp16_gpu",
        required=False,
        action="store_true",
        help="Generate fp16 onnx model for Nvidia GPUs",
    )

    parser.add_argument(
        "--int4_gpu",
        required=False,
        action="store_true",
        help="Generate int4 onnx model for Nvidia GPUs",
    )

    parser.add_argument(
        "--fp16_a100",
        required=False,
        action="store_true",
        help="Generate fp16 onnx model for Nvidia A100",
    )

    parser.add_argument(
        "--int4_a100",
        required=False,
        action="store_true",
        help="Generate int4 onnx model for Nvidia A100",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite existing onnx models",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    converter = ConvertPhi2ToONNX(device)

    temp_onnx_path = "phi2_temp.onnx"
    original_onnx_path = "phi2.onnx"

    if not os.path.exists(original_onnx_path) or args.overwrite:
        converter.dynamo_export(temp_onnx_path)
        converter.preprocess_onnx(
            temp_onnx_path,
            original_onnx_path,
            func_name="modeling_phi_PhiModel_model_1",  # The function to unroll
            use_gqa=True,
        )
        converter.erase_onnx_model(temp_onnx_path)

    # TODO: support batch export
    if args.fp32_cpu:
        converter.init_attn_type_and_precision(AttentionOpType.MultiHeadAttention, Precision.FLOAT32)
        converter.optimize_phi2_onnx(original_onnx_path, "fp32_cpu/phi2_opt.onnx")
    elif args.int4_cpu:
        converter.init_attn_type_and_precision(AttentionOpType.MultiHeadAttention, Precision.INT4)
        converter.optimize_phi2_onnx(original_onnx_path, "int4_cpu/phi2_opt.onnx")
    elif args.fp32_gpu:
        converter.init_attn_type_and_precision(AttentionOpType.Attention, Precision.FLOAT32)
        converter.optimize_phi2_onnx(original_onnx_path, "fp32_gpu/phi2_opt.onnx")
    elif args.fp16_gpu:
        converter.init_attn_type_and_precision(AttentionOpType.Attention, Precision.FLOAT16)
        converter.optimize_phi2_onnx(original_onnx_path, "fp16_gpu/phi2_opt.onnx")
    elif args.int4_gpu:
        converter.init_attn_type_and_precision(AttentionOpType.Attention, Precision.INT4)
        converter.optimize_phi2_onnx(original_onnx_path, "int4_gpu/phi2_opt.onnx")
    elif args.fp16_a100:
        converter.init_attn_type_and_precision(AttentionOpType.GroupQueryAttention, Precision.FLOAT16)
        converter.optimize_phi2_onnx(original_onnx_path, "fp16_a100/phi2_opt.onnx")
    elif args.int4_a100:
        converter.init_attn_type_and_precision(AttentionOpType.GroupQueryAttention, Precision.INT4)
        converter.optimize_phi2_onnx(original_onnx_path, "int4_a100/phi2_opt.onnx")
    else:
        print(
            "Please specify a valid option from --fp32_cpu, --int4_cpu, --fp32_gpu, --fp16_gpu, --int4_gpu, --fp16_a100, --int4_a100"
        )
        return

    # converter.erase_onnx_model(original_onnx_path)
    print("done")


if __name__ == "__main__":
    main()
