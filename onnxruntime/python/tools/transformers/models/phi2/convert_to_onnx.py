# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from itertools import chain
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import torch
import onnx
from transformers import AutoConfig, AutoModelForCausalLM

import sys, os

sys.path.append(os.path.dirname(__file__))

transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if transformers_dir not in sys.path:
    sys.path.append(transformers_dir)


class ConvertPhi2ToONNX:
    def __init__(self, model_class: str, device: torch.device, cache_dir: str = "./cache"):
        self.model_class = model_class
        self.device = device
        self.cache_dir = cache_dir
        self.phi_config = None
        self.phi_model = None
        self.batch_size = 2
        self.sequence_length = 8

    def __inline_function(self, model: onnx.ModelProto, func_name: str) -> onnx.ModelProto:
        """
        Inlines the function with the given name in the model.
        """
        for node in [node for node in model.graph.node if node.op_type == func_name]:
            model.graph.node.remove(node)
        for node in [node for f in model.functions if f.name == func_name for node in f.node]:
            model.graph.node.append(node)

        return model

    def __get_phi2_torch_model(self):
        if self.phi_config is not None and self.phi_model is not None:
            return
        self.phi_config = AutoConfig.from_pretrained(self.model_class, trust_remote_code=True, cache_dir=self.cache_dir)
        self.phi_model = AutoModelForCausalLM.from_pretrained(
            self.model_class, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.phi_model.eval()
        self.phi_model.to(self.device)

    def get_phi2_torch_inputs(self, batch_size: int, sequence_length: int):
        self.__get_phi2_torch_model()
        input_ids = torch.randint(
            low=0, high=self.phi_config.vocab_size, size=(batch_size, sequence_length), dtype=torch.int64, device=device
        )
        torch_inputs = self.phi_model.prepare_inputs_for_generation(
            input_ids, past_key_values=self.phi_model(input_ids, use_cache=True)["past_key_values"]
        )
        return torch_inputs["input_ids"], torch_inputs["attention_mask"], torch_inputs["past_key_values"]

    def erase_onnx_model(self, onnx_path: str):
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

    def inline_onnx(self, onnx_path_in: str, onnx_path_out: str, function_names: List[str]):
        model = onnx.load_model(onnx_path_in, load_external_data=True)
        for function_name in function_names:
            model = self.__inline_function(model, function_name)
        onnx.save_model(
            model,
            onnx_path_out,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=onnx_path_out + ".data",
        )

model_class = "microsoft/phi-2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

converter = ConvertPhi2ToONNX(model_class, device)
converter.dynamo_export("phi-2_temp.onnx")
converter.inline_onnx(
    "phi-2_temp.onnx",
    "phi-2.onnx",
    ["transformers_modules_microsoft_phi-2_85d00b03fee509307549d823fdd095473ba5197c_modeling_phi_PhiModel_model_1"],
)
converter.erase_onnx_model("phi-2_temp.onnx")
converter.erase_onnx_model("phi-2.onnx")

# from fusion_options import FusionOptions
# from optimizer import optimize_model

# output_path = "phi-2_decoder_fp32_opt.onnx"
# optimization_options = FusionOptions("gpt2")
# model_opt = optimize_model(
#     temp_path,
#     model_type="phi",
#     num_heads=phi_config.num_attention_heads,
#     hidden_size=phi_config.hidden_size,
#     opt_level=0,
#     optimization_options=optimization_options,
#     only_onnxruntime=False,
# )
# model_opt.save_model_to_file(output_path, use_external_data_format=True)
