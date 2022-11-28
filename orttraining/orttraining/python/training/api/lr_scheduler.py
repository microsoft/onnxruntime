# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# lr_scheduler.py

from onnxruntime.capi import _pybind_state as C


class LRScheduler:
    """
    Class for running Learning Scheduler Step in Training.
    This class is a wrapper of LRScheduler Class.
    """

    def __init__(self, optimizer, warmup_step_count, total_step_count) -> None:
        """
        Initializes LRScheduler with the optimizer model and other parameters.
        """
        self._scheduler = C.LRScheduler(optimizer._optimizer, warmup_step_count, total_step_count)

    def step(self):
        """
        Run LRScheduler Step.
        """
        self._scheduler.scheduler_step()
