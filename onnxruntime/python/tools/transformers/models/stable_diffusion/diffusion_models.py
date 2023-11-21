# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from stable_diffusion_tensorrt_txt2img.py in diffusers and TensorRT demo diffusion,
# which has the following license:
#
# Copyright 2023 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import tempfile
from typing import Dict, List, Optional

import onnx
import onnx_graphsurgeon as gs
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from onnx import GraphProto, ModelProto, shape_inference
from ort_optimizer import OrtStableDiffusionOptimizer
from polygraphy.backend.onnx.loader import fold_constants
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from onnxruntime.transformers.onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class TrtOptimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self):
        self.graph.cleanup().toposort()

    def get_optimized_onnx_graph(self):
        return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)

    def infer_shapes(self):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_onnx_path = os.path.join(temp_dir, "model.onnx")
                onnx.save_model(
                    onnx_graph,
                    input_onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                )
                output_onnx_path = os.path.join(temp_dir, "model_with_shape.onnx")
                onnx.shape_inference.infer_shapes_path(input_onnx_path, output_onnx_path)
                onnx_graph = onnx.load(output_onnx_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)


class PipelineInfo:
    def __init__(
        self,
        version: str,
        is_inpaint: bool = False,
        is_refiner: bool = False,
        use_vae=False,
        min_image_size=256,
        max_image_size=1024,
        use_fp16_vae=True,
        use_lcm=False,
    ):
        self.version = version
        self._is_inpaint = is_inpaint
        self._is_refiner = is_refiner
        self._use_vae = use_vae
        self._min_image_size = min_image_size
        self._max_image_size = max_image_size
        self._use_fp16_vae = use_fp16_vae
        self._use_lcm = use_lcm
        if is_refiner:
            assert not use_lcm
            assert self.is_xl()

    def is_inpaint(self) -> bool:
        return self._is_inpaint

    def is_xl(self) -> bool:
        return "xl" in self.version

    def is_xl_base(self) -> bool:
        return self.is_xl() and not self._is_refiner

    def is_xl_refiner(self) -> bool:
        return self.is_xl() and self._is_refiner

    def use_safetensors(self) -> bool:
        return self.is_xl()

    def stages(self) -> List[str]:
        if self.is_xl_base():
            return ["clip", "clip2", "unetxl"] + (["vae"] if self._use_vae else [])

        if self.is_xl_refiner():
            return ["clip2", "unetxl", "vae"]

        return ["clip", "unet", "vae"]

    def vae_scaling_factor(self) -> float:
        return 0.13025 if self.is_xl() else 0.18215

    def vae_torch_fallback(self) -> bool:
        return self.is_xl() and not self._use_fp16_vae

    def custom_fp16_vae(self) -> Optional[str]:
        # For SD XL, use a VAE that fine-tuned to run in fp16 precision without generating NaNs
        return "madebyollin/sdxl-vae-fp16-fix" if self._use_fp16_vae and self.is_xl() else None

    def custom_unet(self) -> Optional[str]:
        return "latent-consistency/lcm-sdxl" if self._use_lcm and self.is_xl_base() else None

    @staticmethod
    def supported_versions(is_xl: bool):
        return ["xl-1.0"] if is_xl else ["1.4", "1.5", "2.0-base", "2.0", "2.1", "2.1-base"]

    def name(self) -> str:
        if self.version == "1.4":
            if self.is_inpaint():
                return "runwayml/stable-diffusion-inpainting"
            else:
                return "CompVis/stable-diffusion-v1-4"
        elif self.version == "1.5":
            if self.is_inpaint():
                return "runwayml/stable-diffusion-inpainting"
            else:
                return "runwayml/stable-diffusion-v1-5"
        elif self.version == "2.0-base":
            if self.is_inpaint():
                return "stabilityai/stable-diffusion-2-inpainting"
            else:
                return "stabilityai/stable-diffusion-2-base"
        elif self.version == "2.0":
            if self.is_inpaint():
                return "stabilityai/stable-diffusion-2-inpainting"
            else:
                return "stabilityai/stable-diffusion-2"
        elif self.version == "2.1":
            return "stabilityai/stable-diffusion-2-1"
        elif self.version == "2.1-base":
            return "stabilityai/stable-diffusion-2-1-base"
        elif self.version == "xl-1.0":
            if self.is_xl_refiner():
                return "stabilityai/stable-diffusion-xl-refiner-1.0"
            else:
                return "stabilityai/stable-diffusion-xl-base-1.0"

        raise ValueError(f"Incorrect version {self.version}")

    def short_name(self) -> str:
        return self.name().split("/")[-1].replace("stable-diffusion", "sd")

    def clip_embedding_dim(self):
        # TODO: can we read from config instead
        if self.version in ("1.4", "1.5"):
            return 768
        elif self.version in ("2.0", "2.0-base", "2.1", "2.1-base"):
            return 1024
        elif self.version in ("xl-1.0") and self.is_xl_base():
            return 768
        else:
            raise ValueError(f"Invalid version {self.version}")

    def clipwithproj_embedding_dim(self):
        if self.version in ("xl-1.0"):
            return 1280
        else:
            raise ValueError(f"Invalid version {self.version}")

    def unet_embedding_dim(self):
        if self.version in ("1.4", "1.5"):
            return 768
        elif self.version in ("2.0", "2.0-base", "2.1", "2.1-base"):
            return 1024
        elif self.version in ("xl-1.0") and self.is_xl_base():
            return 2048
        elif self.version in ("xl-1.0") and self.is_xl_refiner():
            return 1280
        else:
            raise ValueError(f"Invalid version {self.version}")

    def min_image_size(self):
        return self._min_image_size

    def max_image_size(self):
        return self._max_image_size

    def default_image_size(self):
        if self.is_xl():
            return 1024
        if self.version in ("2.0", "2.1"):
            return 768
        return 512


class BaseModel:
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        fp16: bool = False,
        max_batch_size: int = 16,
        embedding_dim: int = 768,
        text_maxlen: int = 77,
    ):
        self.name = self.__class__.__name__

        self.pipeline_info = pipeline_info

        self.model = model
        self.fp16 = fp16
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = pipeline_info.min_image_size()
        self.max_image_shape = pipeline_info.max_image_size()
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_ort_optimizer(self):
        model_name_to_model_type = {
            "CLIP": "clip",
            "UNet": "unet",
            "VAE": "vae",
            "UNetXL": "unet",
            "CLIPWithProj": "clip",
        }
        model_type = model_name_to_model_type[self.name]
        return OrtStableDiffusionOptimizer(model_type)

    def get_model(self):
        return self.model

    def from_pretrained(self, model_class, framework_model_dir, hf_token, subfolder, **kwargs):
        model_dir = os.path.join(framework_model_dir, self.pipeline_info.name(), subfolder)

        if not os.path.exists(model_dir):
            model = model_class.from_pretrained(
                self.pipeline_info.name(),
                subfolder=subfolder,
                use_safetensors=self.pipeline_info.use_safetensors(),
                use_auth_token=hf_token,
                **kwargs,
            ).to(self.device)
            model.save_pretrained(model_dir)
        else:
            print(f"Load {self.name} pytorch model from: {model_dir}")

            model = model_class.from_pretrained(model_dir).to(self.device)
        return model

    def load_model(self, framework_model_dir: str, hf_token: str, subfolder: str):
        pass

    def get_input_names(self) -> List[str]:
        pass

    def get_output_names(self) -> List[str]:
        pass

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        pass

    def get_sample_input(self, batch_size, image_height, image_width) -> tuple:
        pass

    def get_profile_id(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        """For TensorRT EP"""
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)

        profile_id = f"_b_{batch_size}" if static_batch else f"_b_{min_batch}_{max_batch}"

        if self.name != "CLIP":
            if static_image_shape:
                profile_id += f"_h_{image_height}_w_{image_width}"
            else:
                profile_id += f"_h_{min_image_height}_{max_image_height}_w_{min_image_width}_{max_image_width}"

        return profile_id

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        """For TensorRT"""
        pass

    def get_shape_dict(self, batch_size, image_height, image_width):
        pass

    def fp32_input_output_names(self) -> List[str]:
        """For CUDA EP, we export ONNX model with FP32 first, then convert it to mixed precision model.
        This is a list of input or output names that are kept as float32 during converting.
        For the first version, we will use same data type as TensorRT.
        """
        return []

    def optimize_ort(
        self,
        input_onnx_path,
        optimized_onnx_path,
        to_fp16=True,
        fp32_op_list=None,
        optimize_by_ort=True,
        optimize_by_fusion=True,
    ):
        optimizer = self.get_ort_optimizer()
        optimizer.optimize(
            input_onnx_path,
            optimized_onnx_path,
            float16=to_fp16,
            keep_io_types=self.fp32_input_output_names(),
            fp32_op_list=fp32_op_list,
            optimize_by_ort=optimize_by_ort,
            optimize_by_fusion=optimize_by_fusion,
        )

    def optimize_trt(self, input_onnx_path, optimized_onnx_path):
        onnx_graph = onnx.load(input_onnx_path)
        opt = TrtOptimizer(onnx_graph)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.cleanup()
        onnx_opt_graph = opt.get_optimized_onnx_graph()

        if onnx_opt_graph.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
            onnx.save_model(
                onnx_opt_graph,
                optimized_onnx_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )
        else:
            onnx.save(onnx_opt_graph, optimized_onnx_path)

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_image_shape else self.min_image_shape
        max_image_height = image_height if static_image_shape else self.max_image_shape
        min_image_width = image_width if static_image_shape else self.min_image_shape
        max_image_width = image_width if static_image_shape else self.max_image_shape
        min_latent_height = latent_height if static_image_shape else self.min_latent_shape
        max_latent_height = latent_height if static_image_shape else self.max_latent_shape
        min_latent_width = latent_width if static_image_shape else self.min_latent_shape
        max_latent_width = latent_width if static_image_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        max_batch_size,
        embedding_dim: int = 0,
        clip_skip=0,
    ):
        super().__init__(
            pipeline_info,
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim if embedding_dim > 0 else pipeline_info.clip_embedding_dim(),
        )
        self.output_hidden_state = pipeline_info.is_xl()

        # see https://github.com/huggingface/diffusers/pull/5057 for more information of clip_skip.
        # Clip_skip=1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
        self.clip_skip = clip_skip

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        # The exported onnx model has no hidden_state. For SD-XL, We will add hidden_state to optimized onnx model.
        return ["text_embeddings"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_image_shape
        )
        return {
            "input_ids": [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

        if self.output_hidden_state:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return (torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device),)

    def add_hidden_states_graph_output(self, model: ModelProto, optimized_onnx_path, use_external_data_format=False):
        graph: GraphProto = model.graph
        hidden_layers = -1
        for i in range(len(graph.node)):
            for j in range(len(graph.node[i].output)):
                name = graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)

        assert self.clip_skip >= 0 and self.clip_skip < hidden_layers

        node_output_name = f"/text_model/encoder/layers.{hidden_layers - 1 - self.clip_skip}/Add_1_output_0"

        # search the name in outputs of all node
        found = False
        for i in range(len(graph.node)):
            for j in range(len(graph.node[i].output)):
                if graph.node[i].output[j] == node_output_name:
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError("Failed to find hidden_states graph output in clip")

        # Insert a Cast  (fp32 -> fp16) node so that hidden_states has same data type as the first graph output.
        graph_output_name = "hidden_states"
        cast_node = onnx.helper.make_node("Cast", inputs=[node_output_name], outputs=[graph_output_name])
        cast_node.attribute.extend([onnx.helper.make_attribute("to", graph.output[0].type.tensor_type.elem_type)])

        hidden_state = graph.output.add()
        hidden_state.CopyFrom(
            onnx.helper.make_tensor_value_info(
                graph_output_name,
                graph.output[0].type.tensor_type.elem_type,
                ["B", self.text_maxlen, self.embedding_dim],
            )
        )

        onnx_model = OnnxModel(model)
        onnx_model.add_node(cast_node)
        onnx_model.save_model_to_file(optimized_onnx_path, use_external_data_format=use_external_data_format)

    def optimize_ort(
        self,
        input_onnx_path,
        optimized_onnx_path,
        to_fp16=True,
        fp32_op_list=None,
        optimize_by_ort=True,
        optimize_by_fusion=True,
    ):
        optimizer = self.get_ort_optimizer()

        if not self.output_hidden_state:
            optimizer.optimize(
                input_onnx_path,
                optimized_onnx_path,
                float16=to_fp16,
                keep_io_types=[],
                fp32_op_list=fp32_op_list,
                keep_outputs=["text_embeddings"],
                optimize_by_ort=optimize_by_ort,
                optimize_by_fusion=optimize_by_fusion,
            )
        elif optimize_by_fusion:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save to a temporary file so that we can load it with Onnx Runtime.
                logger.info("Saving a temporary model to add hidden_states to graph output ...")
                tmp_model_path = os.path.join(tmp_dir, "model.onnx")

                model = onnx.load(input_onnx_path)
                self.add_hidden_states_graph_output(model, tmp_model_path, use_external_data_format=True)
                optimizer.optimize(
                    tmp_model_path,
                    optimized_onnx_path,
                    float16=to_fp16,
                    keep_io_types=[],
                    fp32_op_list=fp32_op_list,
                    keep_outputs=["text_embeddings", "hidden_states"],
                    optimize_by_ort=optimize_by_ort,
                    optimize_by_fusion=optimize_by_fusion,
                )
        else:  # input is optimized model, there is no need to add hidden states.
            optimizer.optimize(
                input_onnx_path,
                optimized_onnx_path,
                float16=to_fp16,
                keep_io_types=[],
                fp32_op_list=fp32_op_list,
                keep_outputs=["text_embeddings", "hidden_states"],
                optimize_by_ort=optimize_by_ort,
                optimize_by_fusion=optimize_by_fusion,
            )

    def optimize_trt(self, input_onnx_path, optimized_onnx_path):
        onnx_graph = onnx.load(input_onnx_path)
        opt = TrtOptimizer(onnx_graph)
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.select_outputs([0], names=["text_embeddings"])  # rename network output
        opt.cleanup()
        onnx_opt_graph = opt.get_optimized_onnx_graph()
        if self.output_hidden_state:
            self.add_hidden_states_graph_output(onnx_opt_graph, optimized_onnx_path)
        else:
            onnx.save(onnx_opt_graph, optimized_onnx_path)

    def load_model(self, framework_model_dir, hf_token, subfolder="text_encoder"):
        return self.from_pretrained(CLIPTextModel, framework_model_dir, hf_token, subfolder)


class CLIPWithProj(CLIP):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        max_batch_size=16,
        clip_skip=0,
    ):
        super().__init__(
            pipeline_info,
            model,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=pipeline_info.clipwithproj_embedding_dim(),
            clip_skip=clip_skip,
        )

    def load_model(self, framework_model_dir, hf_token, subfolder="text_encoder_2"):
        return self.from_pretrained(CLIPTextModelWithProjection, framework_model_dir, hf_token, subfolder)

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.embedding_dim),
        }

        if self.output_hidden_state:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output


class UNet(BaseModel):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        fp16=False,  # used by TRT
        max_batch_size=16,
        text_maxlen=77,
        unet_dim=4,
    ):
        super().__init__(
            pipeline_info,
            model=model,
            device=device,
            fp16=fp16,
            max_batch_size=max_batch_size,
            embedding_dim=pipeline_info.unet_embedding_dim(),
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim

    def load_model(self, framework_model_dir, hf_token, subfolder="unet"):
        options = {"variant": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}
        return self.from_pretrained(UNet2DConditionModel, framework_model_dir, hf_token, subfolder, **options)

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)
        return {
            "sample": [
                (2 * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (2 * batch_size, self.unet_dim, latent_height, latent_width),
                (2 * max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (2 * min_batch, self.text_maxlen, self.embedding_dim),
                (2 * batch_size, self.text_maxlen, self.embedding_dim),
                (2 * max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": [1],
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.tensor([1.0], dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        )

    def fp32_input_output_names(self) -> List[str]:
        return ["sample", "timestep"]


class UNetXL(BaseModel):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        fp16=False,  # used by TRT
        max_batch_size=16,
        text_maxlen=77,
        unet_dim=4,
        time_dim=6,
    ):
        super().__init__(
            pipeline_info,
            model,
            device=device,
            fp16=fp16,
            max_batch_size=max_batch_size,
            embedding_dim=pipeline_info.unet_embedding_dim(),
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.time_dim = time_dim

        self.custom_unet = pipeline_info.custom_unet()
        self.do_classifier_free_guidance = not (self.custom_unet and "lcm" in self.custom_unet)
        self.batch_multiplier = 2 if self.do_classifier_free_guidance else 1

    def load_model(self, framework_model_dir, hf_token, subfolder="unet"):
        options = {"variant": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}

        if self.custom_unet:
            model_dir = os.path.join(framework_model_dir, self.custom_unet, subfolder)
            if not os.path.exists(model_dir):
                unet = UNet2DConditionModel.from_pretrained(self.custom_unet, **options)
                unet.save_pretrained(model_dir)
            else:
                unet = UNet2DConditionModel.from_pretrained(model_dir, **options)
            return unet.to(self.device)

        return self.from_pretrained(UNet2DConditionModel, framework_model_dir, hf_token, subfolder, **options)

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        if self.do_classifier_free_guidance:
            return {
                "sample": {0: "2B", 2: "H", 3: "W"},
                "encoder_hidden_states": {0: "2B"},
                "latent": {0: "2B", 2: "H", 3: "W"},
                "text_embeds": {0: "2B"},
                "time_ids": {0: "2B"},
            }
        return {
            "sample": {0: "B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "B"},
            "latent": {0: "B", 2: "H", 3: "W"},
            "text_embeds": {0: "B"},
            "time_ids": {0: "B"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)
        m = self.batch_multiplier
        return {
            "sample": [
                (m * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (m * batch_size, self.unet_dim, latent_height, latent_width),
                (m * max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (m * min_batch, self.text_maxlen, self.embedding_dim),
                (m * batch_size, self.text_maxlen, self.embedding_dim),
                (m * max_batch, self.text_maxlen, self.embedding_dim),
            ],
            "text_embeds": [(m * min_batch, 1280), (m * batch_size, 1280), (m * max_batch, 1280)],
            "time_ids": [
                (m * min_batch, self.time_dim),
                (m * batch_size, self.time_dim),
                (m * max_batch, self.time_dim),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        m = self.batch_multiplier
        return {
            "sample": (m * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (1,),
            "encoder_hidden_states": (m * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (m * batch_size, 4, latent_height, latent_width),
            "text_embeds": (m * batch_size, 1280),
            "time_ids": (m * batch_size, self.time_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        m = self.batch_multiplier
        return (
            torch.randn(
                m * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.tensor([1.0], dtype=torch.float32, device=self.device),
            torch.randn(m * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            {
                "added_cond_kwargs": {
                    "text_embeds": torch.randn(m * batch_size, 1280, dtype=dtype, device=self.device),
                    "time_ids": torch.randn(m * batch_size, self.time_dim, dtype=dtype, device=self.device),
                }
            },
        )

    def fp32_input_output_names(self) -> List[str]:
        return ["sample", "timestep"]


# VAE Decoder
class VAE(BaseModel):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        max_batch_size,
        fp16: bool = False,
        custom_fp16_vae: Optional[str] = None,
    ):
        super().__init__(
            pipeline_info,
            model=model,
            device=device,
            fp16=fp16,
            max_batch_size=max_batch_size,
        )

        # For SD XL, need custom trained fp16 model to speed up, and avoid overflow at the same time.
        self.custom_fp16_vae = custom_fp16_vae

    def load_model(self, framework_model_dir, hf_token: Optional[str] = None, subfolder: str = "vae_decoder"):
        model_name = self.custom_fp16_vae or self.pipeline_info.name()

        model_dir = os.path.join(framework_model_dir, model_name, subfolder)
        if not os.path.exists(model_dir):
            if self.custom_fp16_vae:
                vae = AutoencoderKL.from_pretrained(self.custom_fp16_vae, torch_dtype=torch.float16).to(self.device)
            else:
                vae = AutoencoderKL.from_pretrained(
                    self.pipeline_info.name(),
                    subfolder="vae",
                    use_safetensors=self.pipeline_info.use_safetensors(),
                    use_auth_token=hf_token,
                ).to(self.device)
            vae.save_pretrained(model_dir)
        else:
            print(f"Load {self.name} pytorch model from: {model_dir}")
            if self.custom_fp16_vae:
                vae = AutoencoderKL.from_pretrained(model_dir, torch_dtype=torch.float16).to(self.device)
            else:
                vae = AutoencoderKL.from_pretrained(model_dir).to(self.device)

        vae.forward = vae.decode
        return vae

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "8H", 3: "8W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return (torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device),)

    def fp32_input_output_names(self) -> List[str]:
        return [] if self.fp16 else ["latent", "images"]


def get_tokenizer(pipeline_info: PipelineInfo, framework_model_dir, hf_token, subfolder="tokenizer"):
    tokenizer_dir = os.path.join(framework_model_dir, pipeline_info.name(), subfolder)

    if not os.path.exists(tokenizer_dir):
        model = CLIPTokenizer.from_pretrained(
            pipeline_info.name(),
            subfolder=subfolder,
            use_safetensors=pipeline_info.is_xl(),
            use_auth_token=hf_token,
        )
        model.save_pretrained(tokenizer_dir)
    else:
        print(f"[I] Load tokenizer pytorch model from: {tokenizer_dir}")
        model = CLIPTokenizer.from_pretrained(tokenizer_dir)
    return model


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae_encoder):
        super().__init__()
        self.vae_encoder = vae_encoder

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoder(BaseModel):
    def __init__(self, pipeline_info: PipelineInfo, model, device, max_batch_size):
        super().__init__(
            pipeline_info,
            model=model,
            device=device,
            max_batch_size=max_batch_size,
        )

    def load_model(self, framework_model_dir, hf_token, subfolder="vae_encoder"):
        vae = self.from_pretrained(AutoencoderKL, framework_model_dir, hf_token, subfolder)
        return TorchVAEEncoder(vae)

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {"images": {0: "B", 2: "8H", 3: "8W"}, "latent": {0: "B", 2: "H", 3: "W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        self.check_dims(batch_size, image_height, image_width)

        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)
