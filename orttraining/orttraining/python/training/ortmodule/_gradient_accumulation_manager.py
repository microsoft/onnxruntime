# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from . import _utils
from onnxruntime.capi import _pybind_state as C


class GradientAccumulationManager(object):
    """Handles Gradient accumulation optimization during training

    This feature must be enabled once before training and cannot be turned off within a training run.
    """
    # TODO: enable switching the feature on/off in the middle of the training

    def __init__(self):
        self.cache = None
        self._param_name_value_map = None
        self._param_version_map = None
        self._frontier_node_arg_map = None
        self._enabled = False
        self._update_cache = False

    def initialize(self, enabled, module, graph_info) -> None:
        """Initializes Gradient Accumulation optimization.

        Args:
            enabled (bool): Whether the optimization is enabled or disabled.
            module (torch.nn.Module): Training model
            graph_info (GraphInfo): The ORT Graph Info object holding information about backend graph.
        """
        if enabled:
            self._enabled = True
            self.cache = C.OrtValueCache()
            # Since named_parameters() is a generator function, need to avoid overhead and
            # populate the params in memory to avoid generating the param map every
            # step. This will not work if the user adds or removes params between steps
            self._param_name_value_map = {
                name: param for name, param in module.named_parameters()}
            self._param_version_map = dict()
            self._frontier_node_arg_map = graph_info.frontier_node_arg_map
            self._cached_node_arg_names = graph_info.cached_node_arg_names
            self._cache_start = len(graph_info.user_output_names)

    @property
    def enabled(self):
        """Indicates whether gradient accumulation optimization is enabled.
        """
        return self._enabled

    def extract_outputs_and_maybe_update_cache(self, forward_outputs):
        """Extract the user outputs from the forward outputs as torch tensor and update cache, if needed

        Args:
            forward_outputs (OrtValueVector): List of outputs returned by forward function
        """
        if not self.enabled:
            return tuple(_utils._ortvalue_to_torch_tensor(forward_output) for forward_output in forward_outputs)
        if self._update_cache:
            for i in range(self._cache_start, len(forward_outputs)):
                self.cache.insert(
                    self._cached_node_arg_names[i-self._cache_start], forward_outputs[i])
            self._update_cache = False
        return tuple(_utils._ortvalue_to_torch_tensor(forward_outputs[i]) for i in range(self._cache_start))

    def maybe_update_cache_before_run(self):
        """Update cache when model parameters are modified and optimization is enabled.
        """
        # The current implementation relies on param._version, which might not be
        # updated in all cases(eg. inplace update)
        # TODO: Make detection of parameter update robust
        if not self.enabled:
            return

        # parse param versions to detect change or no change
        for name, arg_name in self._frontier_node_arg_map.items():
            param = self._param_name_value_map[name]
            if name not in self._param_version_map:
                self._param_version_map[name] = param._version
            elif param._version != self._param_version_map[name]:
                # there is an updated param, so remove entry from cache
                # in order to recompute the value
                if self.cache.count(arg_name):
                    self.cache.remove(arg_name)
                self._update_cache = True
                self._param_version_map[name] = param._version
