# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import tempfile
from pathlib import Path
from typing import List

import onnx
import torch
from onnx_model import OnnxModel
from transformers import WhisperConfig
from whisper_inputs import get_model_dynamic_axes, get_sample_encoder_inputs

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


class WhisperEncoder(torch.nn.Module):
    """Whisper encoder component"""

    def __init__(self, config: WhisperConfig, model: torch.nn.Module, model_impl: str):
        super().__init__()
        self.config = config
        self.device = model.device
        self.model_impl = model_impl

        self.encoder = model.encoder if model_impl == "openai" else model.model.encoder

    def forward(self, audio_features: torch.Tensor):
        outputs = self.encoder(audio_features)
        return outputs if self.model_impl == "openai" else outputs.last_hidden_state

    def input_names(self):
        input_names = ["audio_features"]
        return input_names

    def output_names(self):
        output_names = ["encoder_hidden_states"]
        return output_names

    def dynamic_axes(self, input_names, output_names):
        dynamic_axes = get_model_dynamic_axes(self.config, input_names, output_names)
        return dynamic_axes

    def export_onnx(
        self,
        onnx_model_path: str,
        provider: str,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_fp16_inputs: bool = False,
    ):
        """Export encoder to ONNX

        Args:
            onnx_model_path (str): path to save ONNX model
            provider (str): provider to use for verifying parity on ONNX model
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
            use_fp16_inputs (bool, optional): use float16 inputs for the audio_features. Defaults to False.
        """
        # Shape of encoder's tensors:
        # Inputs:
        #    audio_features: (batch_size, num_mels, num_frames)
        # Outputs:
        #    encoder_hidden_states: (batch_size, num_frames // 2, hidden_size)

        inputs = get_sample_encoder_inputs(
            self.config,
            self.device,
            batch_size=2,
            use_fp16=use_fp16_inputs,
        )

        input_names = self.input_names()
        output_names = self.output_names()
        dynamic_axes = self.dynamic_axes(input_names, output_names)

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            temp_onnx_model_path = os.path.join(tmp_dir_name, "encoder.onnx")
            Path(temp_onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
            out_path = temp_onnx_model_path if use_external_data_format else onnx_model_path

            torch.onnx.export(
                self,
                args=(inputs["audio_features"]),
                f=out_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
                verbose=verbose,
            )

            if use_external_data_format:
                model = onnx.load_model(out_path, load_external_data=use_external_data_format)
                OnnxModel.save(
                    model,
                    onnx_model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                )

        self.verify_onnx(onnx_model_path, provider, use_fp16_inputs)

    def verify_onnx(
        self,
        onnx_model_path: str,
        provider: str,
        use_fp16_inputs: bool,
    ):
        """Verify ONNX model outputs and PyTorch model outputs match 

        Args:
            onnx_model_path (str): path to save ONNX model
            provider (str): execution provider for ONNX model
            use_fp16_inputs (bool, optional): use float16 inputs for the audio_features
        """
        # Shape of encoder's tensors:
        # Inputs:
        #    audio_features: (batch_size, num_mels, num_frames)
        # Outputs:
        #    encoder_hidden_states: (batch_size, num_frames // 2, hidden_size)
        inputs = get_sample_encoder_inputs(
            self.config,
            self.device,
            batch_size=2,
            use_fp16=use_fp16_inputs,
        )

        # Run PyTorch model
        pt_outputs = self.forward(inputs["audio_features"]).detach().cpu().numpy()

        # Run ONNX model
        sess = ort.InferenceSession(onnx_model_path, providers=[provider])
        ort_outputs = sess.run(None, {"audio_features": inputs["audio_features"].detach().cpu().numpy()})[0]
        
        # Calculate output difference
        diff = np.abs(pt_outputs - ort_outputs)
        logger.warning("Comparing encoder_hidden_states...")
        logger.warning(f"Max diff: {np.max(diff)}")
