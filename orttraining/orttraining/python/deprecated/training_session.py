#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import sys
import os

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import Session, InferenceSession, IOBinding

file_store = None

def store_set(store_key, store_value):
    file_store.set(store_key, store_value)

def store_get(store_key):
    return file_store.get(store_key)

class TrainingSession(InferenceSession):
    def __init__(self, path_or_bytes, parameters, sess_options=None):
        Session.__init__(self)

        import torch
        file_store = torch.distributed.FileStore('/bert_ort/liqun/test_out/file_store.dat', parameters.world_size)
        file_store.set("local_rank", str(parameters.world_rank))
        file_store.set("world_rank", str(parameters.world_rank))
        file_store.set("world_size", str(parameters.world_size))

        if sess_options:
            self._sess = C.TrainingSession(sess_options, file_store)
        else:
            self._sess = C.TrainingSession()

        if isinstance(path_or_bytes, str):
            config_result = self._sess.load_model(path_or_bytes, parameters)
        elif isinstance(path_or_bytes, bytes):
            config_result = self._sess.read_bytes(path_or_bytes, parameters)
        else:
            raise TypeError("Unable to load from type '{0}'".format(type(path_or_bytes)))

        self.loss_scale_input_name = config_result.loss_scale_input_name

        self._inputs_meta = self._sess.inputs_meta
        self._outputs_meta = self._sess.outputs_meta

    def __del__(self):
        if self._sess:
            self._sess.finalize()

    def get_state(self):
        return self._sess.get_state()

    def load_state(self, dict, strict=False):
        self._sess.load_state(dict, strict)

    def is_output_fp32_node(self, output_name):
        return self._sess.is_output_fp32_node(output_name)
