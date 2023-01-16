# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# lr_scheduler.py

from onnxruntime.capi import _pybind_state as C


class LinearLRScheduler:
    """
    Linearly updates the learning rate in the optimizer

    The linear learning rate scheduler decays the learning rate by linearly updated
    multiplicative factor from the initial learning rate set on the training session to 0. The decay
    is performed after the initial warm up phase where the learning rate is linearly incremented
    from to the initial learning rate provided.

    Args:
        optimizer (:obj:`training_api.Optimizer`): User's onnxruntime training Optimizer
        warmup_step_count (int): The number of steps in the warm up phase.
        total_step_count (int): The total number of training steps.
        initial_lr (float): The initial learning rate.
    """

    def __init__(self, optimizer, warmup_step_count, total_step_count, initial_lr) -> None:

        self._scheduler = C.LinearLRScheduler(optimizer._optimizer, warmup_step_count, total_step_count, initial_lr)

    def step(self):
        """
        The step method of the LinearLRScheduler class is used to update the learning rate of the optimizer according
        to the scheduler's strategy.
        This method should be called at each step of training to ensure that the learning rate is properly adjusted.
        """
        self._scheduler.scheduler_step()
