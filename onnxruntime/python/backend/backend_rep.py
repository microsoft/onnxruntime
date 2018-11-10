#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
#--------------------------------------------------------------------------
"""
Implements ONNX's backend API.
"""
import numpy as np
from onnx.backend.base import BackendRep


class OnnxRuntimeBackendRep(BackendRep):
    """
    Computes the prediction for a pipeline converted into
    an :class:`onnxruntime.InferenceSession` node.
    """
    
    def __init__(self, session):
        """
        :param session: :class:`onnxruntime.InferenceSession`
        """
        self._session = session
    
    
    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        """
        Computes the prediction.
        See :meth:`onnxruntime.InferenceSession.run`.
        """
        if isinstance(inputs, list):
            inps = {}
            for i, inp in enumerate(self._session.get_inputs()):
                inps[inp.name] = inputs[i]
            outs = self._session.run(None, inps)
            if isinstance(outs, list):
                return outs
            else:
                output_names = [o.name for o in self._session.get_outputs()]
                return [outs[name] for name in output_names]
        else:
            inp = self._session.get_inputs()
            if len(inp) != 1:
                raise RuntimeError("Model expect {0} inputs".format(len(inp)))
            inps = {inp[0].name: inputs}
            return self._session.run(None, inps)
