# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from onnxruntime.capi import _pybind_state as C
from onnxruntime.training.api.optimizer import Optimizer


class LinearLRScheduler:
    """Linearly updates the learning rate in the optimizer

    The linear learning rate scheduler decays the learning rate by linearly updated
    multiplicative factor from the initial learning rate set on the training session to 0. The decay
    is performed after the initial warm up phase where the learning rate is linearly incremented
    from 0 to the initial learning rate provided.

    Args:
        optimizer: User's onnxruntime training Optimizer
        warmup_step_count: The number of steps in the warm up phase.
        total_step_count: The total number of training steps.
        initial_lr: The initial learning rate.
    """

    def __init__(self, optimizer: Optimizer, warmup_step_count: int, total_step_count: int, initial_lr: float):
        self._scheduler = C.LinearLRScheduler(optimizer._optimizer, warmup_step_count, total_step_count, initial_lr)

    def step(self) -> None:
        """Updates the learning rate of the optimizer linearly.

        This method should be called at each step of training to ensure that the learning rate is properly adjusted.
        """
        self._scheduler.scheduler_step()
