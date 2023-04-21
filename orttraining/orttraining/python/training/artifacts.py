# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import pathlib
from enum import Enum
from typing import List, Optional, Union

import onnx

import onnxruntime.training.onnxblock as onnxblock


class LossType(Enum):
    """Enum to represent the loss functions supported by ORT

    To be used with the `loss` parameter of `generate_artifacts` function.
    """

    MSELoss = 1
    CrossEntropyLoss = 2
    BCEWithLogitsLoss = 3
    L1Loss = 4


class OptimType(Enum):
    """Enum to represent the optimizers supported by ORT

    To be used with the `optimizer` parameter of `generate_artifacts` function.
    """

    AdamW = 1


def generate_artifacts(
    model: onnx.ModelProto,
    requires_grad: Optional[List[str]] = None,
    frozen_params: Optional[List[str]] = None,
    loss: Optional[Union[LossType, onnxblock.Block]] = None,
    optimizer: Optional[OptimType] = None,
    artifact_directory: Optional[Union[str, bytes, os.PathLike]] = None,
    **extra_options,
) -> None:
    """Generates artifacts required for training with ORT training api.

    This function generates the following artifacts:
        1. Training model (onnx.ModelProto): Contains the base model graph, loss sub graph and the gradient graph.
        2. Eval model (onnx.ModelProto):  Contains the base model graph and the loss sub graph
        3. Checkpoint: Contains the model parameters.
        4. Optimizer model (onnx.ModelProto): Model containing the optimizer graph.

    Args:
        model: The base model to be used for gradient graph generation.
        requires_grad: List of names of model parameters that require gradient computation
        frozen_params: List of names of model parameters that should be frozen.
        loss: The loss function enum to be used for training. If None, no loss node is added to the graph.
        optimizer: The optimizer enum to be used for training. If None, no optimizer model is generated.
        artifact_directory: The directory to save the generated artifacts.
                                          If None, the current working directory is used.
        **extra_options: Additional keyword arguments for artifact generation.
            prefix: The prefix to be used for the generated artifacts. If not specified, no prefix is used.

    Raises:
        RuntimeError: If the loss provided is not one of the supported losses or an instance of onnxblock.Block.
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
        def __init__(self, _loss):
            super().__init__()
            self._loss = _loss

        def build(self, *inputs_to_loss):
            return self._loss(*inputs_to_loss)

    training_block = _TrainingBlock(loss_block)

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
    with onnxblock.base(model):
        _ = training_block(*[output.name for output in model.graph.output])
        training_model, eval_model = training_block.to_model_proto()
        model_params = training_block.parameters()

    if artifact_directory is None:
        artifact_directory = pathlib.Path.cwd()
    prefix = ""
    if "prefix" in extra_options:
        prefix = extra_options["prefix"]
        logging.info("Using prefix %s for generated artifacts.", prefix)

    artifact_directory = pathlib.Path(artifact_directory)

    training_model_path = artifact_directory / f"{prefix}training_model.onnx"
    if os.path.exists(training_model_path):
        logging.info("Training model path %s already exists. Overwriting.", training_model_path)
    onnx.save(training_model, training_model_path)
    logging.info("Saved training model to %s", training_model_path)

    eval_model_path = artifact_directory / f"{prefix}eval_model.onnx"
    if os.path.exists(eval_model_path):
        logging.info("Eval model path %s already exists. Overwriting.", eval_model_path)
    onnx.save(eval_model, eval_model_path)
    logging.info("Saved eval model to %s", eval_model_path)

    checkpoint_path = artifact_directory / f"{prefix}checkpoint"
    if os.path.exists(checkpoint_path):
        logging.info("Checkpoint path %s already exists. Overwriting.", checkpoint_path)
    onnxblock.save_checkpoint(training_block.parameters(), checkpoint_path)
    logging.info("Saved checkpoint to %s", checkpoint_path)

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

    optim_model = None
    optim_blocks = {OptimType.AdamW: onnxblock.optim.AdamW}
    optim_block = optim_blocks[optimizer]()
    with onnxblock.empty_base():
        _ = optim_block(model_params)
        optim_model = optim_block.to_model_proto()

    optimizer_model_path = artifact_directory / f"{prefix}optimizer_model.onnx"
    onnx.save(optim_model, optimizer_model_path)
    logging.info("Saved optimizer model to %s", optimizer_model_path)
