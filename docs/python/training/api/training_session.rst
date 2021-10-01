
Training API
============

.. contents::
    :local:

Options and Parameters
++++++++++++++++++++++

TrainingParameters
^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.TrainingParameters
    :members:
    :inherited-members:
    :undoc-members:

Hidden API
++++++++++

GraphInfo
^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.GraphInfo
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

GradientNodeAttributeDefinition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.GradientNodeAttributeDefinition
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

GradientNodeDefinition
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.GradientNodeDefinition
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

GraphTransformerConfiguration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.GraphTransformerConfiguration
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

OrtModuleGraphBuilder
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.OrtModuleGraphBuilder
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

OrtModuleGraphBuilderConfiguration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.OrtModuleGraphBuilderConfiguration
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

OrtValueCache
^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.OrtValueCache
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

OrtValueVector
^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.OrtValueVector
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

PartialGraphExecutionState
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.PartialGraphExecutionState
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

PropagateCastOpsConfiguration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.PropagateCastOpsConfiguration
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

TrainingConfigurationResult
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.TrainingConfigurationResult
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:
    
TrainingGraphTransformerConfiguration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.capi._pybind_state.TrainingGraphTransformerConfiguration
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :undoc-members:

Functions
+++++++++

.. autofunc:: onnxruntime.capi._pybind_state.register_aten_op_executor

.. autofunc:: onnxruntime.capi._pybind_state.register_backward_runner

.. autofunc:: onnxruntime.capi._pybind_state.register_forward_runner

.. autofunc:: onnxruntime.capi._pybind_state.register_torch_autograd_function

.. autofunc:: onnxruntime.capi._pybind_state.register_gradient_definition

.. autofunc:: onnxruntime.capi._pybind_state.unregister_python_functions

TrainingSession
+++++++++++++++

.. autoclass:: onnxruntime.TrainingSession
    :members: get_state, get_model_state, get_optimizer_state, get_partition_info_map, load_state, is_output_fp32_node
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
