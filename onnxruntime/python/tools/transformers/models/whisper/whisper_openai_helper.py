# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy
import onnx
import torch
from io_binding_helper import TypeHelper
from models.t5.past_helper import PastKeyValuesHelper
from onnx_model import OnnxModel
from torch_onnx_export_helper import torch_onnx_export
from transformers import WhisperConfig, file_utils

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


class WhisperDecoderInitOpenai(torch.nn.Module):
    """WhisperDecoderInit."""
    def __init__(
        self,
        model: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        super().__init__()
        self.whisper_model = model
        self.whisper_decoder = decoder
        self.kv_cache = {}

    @torch.no_grad()
    def forward(
        self,
        tokens,
        audio_features,
    ):
        if not self.kv_cache:
            self.kv_cache, _ = self.whisper_model.install_kv_cache_hooks()
        logits = self.whisper_decoder(tokens, audio_features, kv_cache=self.kv_cache)
        return logits, list(self.kv_cache.values())
