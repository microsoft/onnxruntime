# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Implements ONNX's backend API.
"""

from onnx.backend.base import BackendRep

from onnxruntime import RunOptions

# Allowlist of RunOptions attributes that are safe to set via the backend API.
# 'terminate' excluded: setting it True would deny the current inference call.
# 'training_mode' excluded: silently switches inference behavior in training builds.
_ALLOWED_RUN_OPTIONS = frozenset(
    {
        "log_severity_level",
        "log_verbosity_level",
        "logid",
        "only_execute_path_to_fetches",
    }
)


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

        options = RunOptions()
        for k, v in kwargs.items():
            if k in _ALLOWED_RUN_OPTIONS:
                setattr(options, k, v)
            elif hasattr(options, k):
                raise RuntimeError(
                    f"RunOptions attribute '{k}' is not permitted via the backend API. "
                    f"Allowed attributes: {', '.join(sorted(_ALLOWED_RUN_OPTIONS))}"
                )
            # else: silently ignore unknown keys

        if isinstance(inputs, list):
            inps = {}
            for i, inp in enumerate(self._session.get_inputs()):
                inps[inp.name] = inputs[i]
            outs = self._session.run(None, inps, options)
            if isinstance(outs, list):
                return outs
            else:
                output_names = [o.name for o in self._session.get_outputs()]
                return [outs[name] for name in output_names]
        else:
            inp = self._session.get_inputs()
            if len(inp) != 1:
                raise RuntimeError(f"Model expect {len(inp)} inputs")
            inps = {inp[0].name: inputs}
            return self._session.run(None, inps, options)
