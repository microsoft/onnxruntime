#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import sys
import os

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import Session, InferenceSession, IOBinding


class TrainingSession(InferenceSession):
    def __init__(self, path_or_bytes, parameters, sess_options=None):
        Session.__init__(self)

        if sess_options:
            self._sess = C.TrainingSession(sess_options)
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

    def get_model_state(self, include_mixed_precision_weights=False):
        return self._sess.get_model_state(include_mixed_precision_weights)
    
    def get_optimizer_state(self):
        return self._sess.get_optimizer_state()

    def get_partition_info_map(self):
        return self._sess.get_partition_info_map()

    def load_state(self, dict, strict=False):
        self._sess.load_state(dict, strict)
    
    def load_model_opt_state(self, model_state, opt_state, strict=False):
        self._sess.load_model_opt_state(dict, model_state, opt_state, strict)

    def is_output_fp32_node(self, output_name):
        return self._sess.is_output_fp32_node(output_name)
