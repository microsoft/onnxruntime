#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import sys
import os

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import (Session, InferenceSession, IOBinding,
                                                               check_and_normalize_provider_args)


class TrainingSession(InferenceSession):
    def __init__(self, path_or_bytes, parameters, sess_options=None, providers=None, provider_options=None):
        Session.__init__(self)

        if sess_options:
            self._sess = C.TrainingSession(sess_options)
        else:
            self._sess = C.TrainingSession()

        providers, provider_options = check_and_normalize_provider_args(providers, provider_options,
                                                                        C.get_available_providers())

        if isinstance(path_or_bytes, str):
            config_result = self._sess.load_model(path_or_bytes, parameters, providers, provider_options)
        elif isinstance(path_or_bytes, bytes):
            config_result = self._sess.read_bytes(path_or_bytes, parameters, providers, provider_options)
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
