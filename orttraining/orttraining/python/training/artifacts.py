# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import logging
import os
import pathlib
from enum import Enum
from typing import List, Optional, Union

import onnx

from onnxruntime.tools.convert_onnx_models_to_ort import OptimizationStyle, convert_onnx_models_to_ort
from onnxruntime.training import onnxblock


class LossType(Enum):
    """Loss type to be added to the training model.

    To be used with the `loss` parameter of `generate_artifacts` function.
    """

    MSELoss = 1
    CrossEntropyLoss = 2
    BCEWithLogitsLoss = 3
    L1Loss = 4


class OptimType(Enum):
    """Optimizer type to be to be used while generating the optimizer model for training.

    To be used with the `optimizer` parameter of `generate_artifacts` function.
    """

    AdamW = 1
    SGD = 2


def generate_artifacts(
    model: onnx.ModelProto,
    requires_grad: Optional[List[str]] = None,
    frozen_params: Optional[List[str]] = None,
    loss: Optional[Union[LossType, onnxblock.Block]] = None,
    optimizer: Optional[OptimType] = None,
    artifact_directory: Optional[Union[str, bytes, os.PathLike]] = None,
    prefix: str = "",
    ort_format: bool = False,
    custom_op_library: Optional[Union[str, bytes, os.PathLike]] = None,
    additional_output_names: Optional[List[str]] = None,
    nominal_checkpoint: bool = False,
    loss_input_names: Optional[List[str]] = None,
) -> None:
    """Generates artifacts required for training with ORT training api.

    This function generates the following artifacts:
        1. Training model (onnx.ModelProto): Contains the base model graph, loss sub graph and the gradient graph.
        2. Eval model (onnx.ModelProto):  Contains the base model graph and the loss sub graph
        3. Checkpoint (directory): Contains the model parameters.
        4. Optimizer model (onnx.ModelProto): Model containing the optimizer graph.

    All generated ModelProtos will use the same opsets defined by *model*.

    Args:
        model: The base model to be used for gradient graph generation.
        requires_grad: List of names of model parameters that require gradient computation
        frozen_params: List of names of model parameters that should be frozen.
        loss: The loss function enum to be used for training. If None, no loss node is added to the graph.
        optimizer: The optimizer enum to be used for training. If None, no optimizer model is generated.
        artifact_directory: The directory to save the generated artifacts.
            If None, the current working directory is used.
        prefix: The prefix to be used for the generated artifacts. If not specified, no prefix is used.
        ort_format: Whether to save the generated artifacts in ORT format or not. Default is False.
        custom_op_library: The path to the custom op library.
            If not specified, no custom op library is used.
        additional_output_names: List of additional output names to be added to the training/eval model in addition
            to the loss output. Default is None.
        nominal_checkpoint: Whether to generate the nominal checkpoint in addition to the complete checkpoint.
            Default is False. Nominal checkpoint is a checkpoint that contains nominal information about the model
            parameters. It can be used on the device to reduce overhead while constructing the training model
            as well as to reduce the size of the checkpoint packaged with the on-device application.
        loss_input_names: Specifies a list of input names to be used specifically for the loss computation. When provided,
            only these inputs will be passed to the loss function. If `None`, all graph outputs are passed to
            the loss function.
    Raises:
        RuntimeError: If the loss provided is neither one of the supported losses nor an instance of `onnxblock.Block`
        RuntimeError: If the optimizer provided is not one of the supported optimizers.
    """

    loss_blocks = {
        LossType.MSELoss: onnxblock.loss.MSELoss,
        LossType.CrossEntropyLoss: onnxblock.loss.CrossEntropyLoss,
        LossType.BCEWithLogitsLoss: onnxblock.loss.BCEWithLogitsLoss,
        LossType.L1Loss: onnxblock.loss.L1Loss,
    }

    loss_block = None
    if loss is None:
        loss_block = onnxblock.blocks.PassThrough()
        logging.info("No loss function enum provided. Loss node will not be added to the graph.")
    elif isinstance(loss, LossType):
        loss_block = loss_blocks[loss]()
        logging.info("Loss function enum provided: %s", loss.name)
    else:
        # If a custom implementation of the loss was provided, then it should be
        # accepted and the custom implementation must control the creation of the loss node
        # in the training model.
        # To do this, user must provide an instance of onnxblock.Block.
        if not isinstance(loss, onnxblock.Block):
            raise RuntimeError(
                f"Unknown loss provided {type(loss)}. Expected loss to be either one of"
                "onnxruntime.training.artifacts.LossType or onnxruntime.training.onnxblock.Block."
            )
        loss_block = loss
        logging.info("Custom loss block provided: %s", loss.__class__.__name__)

    class _TrainingBlock(onnxblock.TrainingBlock):
        def __init__(self, _loss, _loss_input_names=None):
            super().__init__()
            self._loss = _loss
            self._loss_input_names = _loss_input_names

        def build(self, *inputs_to_loss):
            # If loss_input_names is passed, only pass the specified input names to the loss function.
            if self._loss_input_names:
                inputs_to_loss = self._loss_input_names

            if additional_output_names:
                # If additional output names is not a list, raise an error
                if not isinstance(additional_output_names, list):
                    raise RuntimeError(
                        f"Unknown type provided for additional output names {type(additional_output_names)}. "
                        "Expected additional output names to be a list of strings."
                    )

                loss_output = self._loss(*inputs_to_loss)
                if isinstance(loss_output, tuple):
                    return (*loss_output, *tuple(additional_output_names))
                else:
                    return (loss_output, *tuple(additional_output_names))

            return self._loss(*inputs_to_loss)

    training_block = _TrainingBlock(loss_block, loss_input_names)

    if requires_grad is not None and frozen_params is not None and set(requires_grad).intersection(set(frozen_params)):
        raise RuntimeError(
            "A parameter cannot be frozen and require gradient computation at the same "
            f"time {set(requires_grad).intersection(set(frozen_params))}"
        )

    if requires_grad is not None:
        for arg in requires_grad:
            training_block.requires_grad(arg)

    if frozen_params is not None:
        for arg in frozen_params:
            training_block.requires_grad(arg, False)

    training_model = None
    eval_model = None
    model_params = None

    custom_op_library_path = None
    if custom_op_library is not None:
        logging.info("Custom op library provided: %s", custom_op_library)
        custom_op_library_path = pathlib.Path(custom_op_library)

    with onnxblock.base(model), (
        onnxblock.custom_op_library(custom_op_library_path)
        if custom_op_library is not None
        else contextlib.nullcontext()
    ):
        _ = training_block(*[output.name for output in model.graph.output])
        training_model, eval_model = training_block.to_model_proto()
        model_params = training_block.parameters()

    def _export_to_ort_format(model_path, output_dir, ort_format, custom_op_library_path):
        if ort_format:
            convert_onnx_models_to_ort(
                model_path,
                output_dir=output_dir,
                custom_op_library_path=custom_op_library_path,
                optimization_styles=[OptimizationStyle.Fixed],
            )

    if artifact_directory is None:
        artifact_directory = pathlib.Path.cwd()
    artifact_directory = pathlib.Path(artifact_directory)

    if prefix:
        logging.info("Using prefix %s for generated artifacts.", prefix)

    training_model_path = artifact_directory / f"{prefix}training_model.onnx"
    if os.path.exists(training_model_path):
        logging.info("Training model path %s already exists. Overwriting.", training_model_path)
    onnx.save(training_model, training_model_path)
    _export_to_ort_format(training_model_path, artifact_directory, ort_format, custom_op_library_path)
    logging.info("Saved training model to %s", training_model_path)

    eval_model_path = artifact_directory / f"{prefix}eval_model.onnx"
    if os.path.exists(eval_model_path):
        logging.info("Eval model path %s already exists. Overwriting.", eval_model_path)
    onnx.save(eval_model, eval_model_path)
    _export_to_ort_format(eval_model_path, artifact_directory, ort_format, custom_op_library_path)
    logging.info("Saved eval model to %s", eval_model_path)

    checkpoint_path = artifact_directory / f"{prefix}checkpoint"
    if os.path.exists(checkpoint_path):
        logging.info("Checkpoint path %s already exists. Overwriting.", checkpoint_path)
    onnxblock.save_checkpoint(training_block.parameters(), checkpoint_path, nominal_checkpoint=False)
    logging.info("Saved checkpoint to %s", checkpoint_path)
    if nominal_checkpoint:
        nominal_checkpoint_path = artifact_directory / f"{prefix}nominal_checkpoint"
        onnxblock.save_checkpoint(training_block.parameters(), nominal_checkpoint_path, nominal_checkpoint=True)
        logging.info("Saved nominal checkpoint to %s", nominal_checkpoint_path)

    # If optimizer is not specified, skip creating the optimizer model
    if optimizer is None:
        logging.info("No optimizer enum provided. Skipping optimizer model generation.")
        return

    if not isinstance(optimizer, OptimType):
        raise RuntimeError(
            f"Unknown optimizer provided {type(optimizer)}. Expected optimizer to be of type "
            "onnxruntime.training.artifacts.OptimType."
        )

    logging.info("Optimizer enum provided: %s", optimizer.name)

    opset_version = None
    for domain in model.opset_import:
        if domain.domain == "" or domain.domain == "ai.onnx":
            opset_version = domain.version
            break

    optim_model = None
    optim_blocks = {OptimType.AdamW: onnxblock.optim.AdamW, OptimType.SGD: onnxblock.optim.SGD}

    optim_block = optim_blocks[optimizer]()
    with onnxblock.empty_base(opset_version=opset_version):
        _ = optim_block(model_params)
        optim_model = optim_block.to_model_proto()

    optimizer_model_path = artifact_directory / f"{prefix}optimizer_model.onnx"
    onnx.save(optim_model, optimizer_model_path)
    _export_to_ort_format(optimizer_model_path, artifact_directory, ort_format, custom_op_library_path)
    logging.info("Saved optimizer model to %s", optimizer_model_path)
