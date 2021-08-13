# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from . import _utils
from onnxruntime.capi import _pybind_state as C

class GradientAccumulationManager(object):
    """
    GradientAccumulationManager is resposible for handling Gradient accumulation optimization
    during training
    """
    def __init__(self):
        """
        :param state: State of partial run that contains intermediate tensors needed to resume the run later.
        :param output_info: Output info.
        """
        self.cache = C.OrtValueCache()
        self._param_name_value_map = None
        self._param_version_map = dict()
        self._frontier_node_arg_map = dict()
        self._enabled = False
        self._update_cache = False

    def initialize(self, module, graph_info) -> None:
        """
         Initialize the Gradient Accumulation optimization.
         :param module: The torch.nn.module which is being trained.
         :param graph_info: The ORT Graph Info object holding information about backend graph.
        """
        # Since named_parameters() is a generator function, need to avoid overhead and
        # populate the params in memory to avoid generating the param map every
        # step. This will not work if the user adds or removes params between steps
        self._param_name_value_map = {name: param for name, param in module.named_parameters()}
        self._frontier_node_arg_map = graph_info.frontier_node_arg_map
        self._cached_node_arg_names = graph_info.cached_node_arg_names
        self._cache_start = len(graph_info.user_output_names)
        self._enabled = True
    
    @property
    def enabled(self):
        """
        This property indicates whether gradient accumulation optimization is enabled.
        """
        return self._enabled

    def update_cache_and_return_outputs(self, forward_outputs):
        """
         Convert the forward outputs to torch and maybe update cache
         :param forward_outputs: OrtValue vector returned by forward function
        """
        if self._update_cache:
            for i in range(self._cache_start, len(forward_outputs)):
                self.cache.insert(self._cached_node_arg_names[i-self._cache_start], forward_outputs[i])
            self._update_cache = False
        return tuple(_utils._ortvalue_to_torch_tensor(forward_outputs[i]) for i in range(self._cache_start))

    def update_cache_before_run(self):
        """
         Iterate over the model parameters to detect whether they have been updated
         and modify the cache accordingly.
        """
        # The current implementation relies on param._version, which might not be
        # updated in all cases(eg. inplace update)
        # TODO: Make detection of parameter update robust
        assert self.enabled, "Gradient accumulation optimization must be enabled by calling " + \
                              "initialize() before this function"    

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
