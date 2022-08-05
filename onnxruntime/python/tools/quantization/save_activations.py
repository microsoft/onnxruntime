# --------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Utilities to run a given ONNX model, while saving input/output tensors of
eligible operator nodes.

A use case is to debug quantization induced accuracy drop. An AI engineer can
run the original float32 model and the quantized model with the same inputs,
then compare the corresponding activations between the two models to find
where the divergence is. 

Example Usage:

.. code-block:: python
    class ExampleDataReader(CalibrationDataReader):
        def __init__(self):
            ...
        def get_next(self):
            ...

    input_data_reader = ExampleDataReader()

    aug_model = aug_model_for_act_saving(path_to_onnx_model)
    augmented_model_path = str(Path(self._tmp_model_dir.name).joinpath("augmented_model.onnx"))
    onnx.save(
        aug_model,
        augmented_model_path,
        save_as_external_data=False,
    )

    tensor_dict = run_collect_activations(augmented_model_path, data_reader)
    # tensor_dict points to a dictionary where the keys are tensor names and each value
    # is a list of tensors, one from each model run

"""

import uuid

from typing import Union,List
from pathlib import Path

import numpy
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper

import onnxruntime
from .calibrate import CalibrationDataReader, CalibraterBase
from .quant_utils import clone_model_with_shape_infer


def aug_model_for_act_saving(
    onnx_model:Union[str,Path,ModelProto],
    op_types_for_saving:List=None
) -> ModelProto:
    r"""Augment a given ONNX model by adding activation tensors as outputs

    Parameters
    ----------
    model : ModelProto or Path
        an ONNX model or the path to load the model
    op_types_for_saving : List
        Optional list of operator types for which the input/output should be
        saved. By default, saving all the float32/float16 tensors.

    Returns
    -------
    Augmented ONNX model
    """

    if op_types_for_saving is None :
        op_types_for_saving = []
    saver = CalibraterBase(onnx_model, op_types_to_calibrate=op_types_for_saving)
    model = clone_model_with_shape_infer(saver.model) # type: ModelProto
    tensors, _ = saver.select_tensors_to_calibrate(model)
    reshape_shape_name = str(uuid.uuid4())
    reshape_shape = numpy_helper.from_array(
        numpy.array([-1], dtype=numpy.int64), reshape_shape_name)
    model.graph.initializer.append(reshape_shape)

    for tensor_name in tensors:
        reshape_output = tensor_name + "_ReshapedSavedOutput"
        reshape_node = onnx.helper.make_node(
            "Reshape",
            inputs=[tensor_name, reshape_shape_name],
            outputs=[reshape_output],
            name=reshape_output,
        )
        model.graph.node.extend([reshape_node])
        vinfo = helper.make_tensor_value_info(
            reshape_output, TensorProto.FLOAT, [1])
        model.graph.output.append(vinfo)
    return model


def run_collect_activations(
    augmented_model:str,
    input_reader: CalibrationDataReader,
    session_options=None,
    execution_providers:List[str]=None
) -> dict:
    r"""Run augmented model and collect activations tensors

    Parameters
    ----------
    augmented_model : str
        Path to augmented model created by aug_model_for_act_saving()
    input_reader : CalibrationDataReader
        Logic for reading input for the model, augmented model have the same
        input with the original model.
    session_options : 
        Optional OnnxRuntime session options for controlling model run.
        By default graph optimization is turned off
    execution_providers : List[str]
        Optional execution providers for running the model. CPU EP is used
        by default.

    Returns
    -------
    A dictionary where the key is tensor name and values are list of tensors from each batch
    """

    if (session_options is None):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    if (execution_providers is None):
        execution_providers = ["CPUExecutionProvider"]

    infer_session = onnxruntime.InferenceSession(
        augmented_model,
        sess_options=session_options,
        providers=execution_providers,
    )
 
    intermediate_outputs = []
    while True:
        inputs = input_reader.get_next()
        if not inputs:
            break
        intermediate_outputs.append(infer_session.run(None, inputs))

        if len(intermediate_outputs) == 0:
            raise ValueError("No data is collected while running augmented model!")

    output_dict={}
    output_info = infer_session.get_outputs()
    for batch in intermediate_outputs:
        for output, output_data in zip(output_info, batch):
            if output.name.endswith("_ReshapedSavedOutput") :
                oname = output.name[0:len(output.name)-len("_ReshapedSavedOutput")]
                output_dict.setdefault(oname, []).append(output_data)

    return output_dict

